print("importing Torch...")
from torch import bfloat16, inference_mode
print("importing Aesthetic Predictor...")
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
print("importing the rest...")
from PIL import Image
import os
from openskill.models import ThurstoneMostellerPart
import math
import matplotlib.pyplot as plt
import shutil
import random
print("Library import completed")

PLAYER_MATCH_LIMIT = 10
BETA = 4.0

#How to make sorting end faster:
#Decrease PLAYER_MATCH_LIMIT (linear hard cap on number of allowed matches)
#Decrease BETA (system converges faster: individual matches are viewed as more informative)

#Get list of images
def load_images_from_folder(folder_path: str) -> list[str]:
    """
    Docstring for load_images_from_folder
    
    :param folder_path: file path to the folder containing the images
    :type folder_path: str
    :return: Return list of image names including their extension
    :rtype: list[str]
    """
    if not os.path.exists(folder_path):
        raise ValueError(f"Folder path does not exist: {folder_path}")
        
    if not os.path.isdir(folder_path):
        raise ValueError(f"Path is not a directory: {folder_path}")
        
    valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    images = []
    
    try:
        for filename in os.listdir(folder_path):
            if any(filename.lower().endswith(ext) for ext in valid_extensions):
                full_path = os.path.join(folder_path, filename)
                #Verify the file can be opened as an image
                try:
                    with Image.open(full_path) as img:
                        img.verify()
                    images.append(filename)
                except Exception as e:
                    print(f"Warning: Skipping {filename} - not a valid image ({e})")
    except PermissionError:
        raise ValueError(f"Permission denied accessing folder: {folder_path}")
    
    if not images:
        print(f"Warning: No valid images found in {folder_path}")
    
    return images

def generate_starting_scores(folder_path: str, images: list[str]) -> list[float]:
    """
    Docstring for generate_starting_scores
    
    :param folder_path: Path to the image folder
    :type folder_path: str
    :param images: List of all image filenames
    :type images: list[str]
    :return: List of seed scores for each image, to allow for slighlty faster score converging
    :rtype: list[float]
    """
    print(f"Rating {folder_path}...")
    ratings = []

    #Load model & preprocessor
    model, preprocessor = convert_v2_5_from_siglip(low_cpu_mem_usage=True, trust_remote_code=True)
    model = model.to(bfloat16).cuda()

    os_path = os.path.abspath(folder_path)
    images = load_images_from_folder(os_path)

    for i in images:
        full_path = os.path.join(os_path, i)
        img = Image.open(full_path).convert("RGB")
        #Preprocess
        pixel_values = (
            preprocessor(images=img, return_tensors="pt")
            .pixel_values.to(bfloat16)
            .cuda()
        )
        #Predict
        with inference_mode():
            score = model(pixel_values).logits.squeeze().float().cpu().numpy()
        ratings.append(float(score*2) + 30) #this IQA generates scores ~5.5

    print("Finished initial rating")
    return ratings

def update_match(model, player_list, idx1, idx2, score1, score2):
    """
    Updates ratings using the degrees of preference (scores).
    score1 and score2 represent the subjective 'strength' of the result.
    Example: 10 vs 0 for a landslide, 6 vs 5 for a near-tie.
    
    :param player_list: list of players
    :param idx1: id of the first player
    :param idx2: id of the second player
    :param score1: Score of the first player in the match played
    :param score2: Score of the second player in the match played
    :param model: the openskill model being used
    """
    p1 = player_list[idx1]
    p2 = player_list[idx2]
    
    teams = [[p1], [p2]]
    new_ratings = model.rate(teams, scores=[score1, score2])
    
    # Update the players in the original list
    player_list[idx1] = new_ratings[0][0]
    player_list[idx2] = new_ratings[1][0]

def select_next_match(player_list: list, seen_matches: set, player_matchcount: list[int], match_limit: int):
    """
    Selects pairs for next match.
    To reduce match count, skip:
        All players below our sigma threshold
        All matchups that have already occured
        All players that have hit a match limit
    
    :param player_list: list of potential players (images)
    :param seen_matches: set of matchups that have already happened
    :param player_matchcount: matchcount for each player
    :param match_limit: maximum matches per player
    """
    best_pair = None
    best_score = -float('inf')

    allowed_ids = [] #List of players not at their match limit
    for i in range(len(player_list)):
        if player_matchcount[i] < match_limit:
            allowed_ids.append(i)

    if len(allowed_ids) <= 1:
        return None
    
    choice = random.random()

    for i in allowed_ids:
        for j in range(i + 1, len(player_list)):
            if player_matchcount[j] >= match_limit:
                continue
            if (i, j) in seen_matches:
                continue
            p1, p2 = player_list[i], player_list[j]

            #Estimate information gain from a given match. Most informative match should be next
            total_var = p1.sigma**2 + p2.sigma**2 + 2 * BETA
            mu_diff = p1.mu - p2.mu
            c = math.sqrt(total_var)
            win_prob = 1.0 / (1.0 + math.exp(-1.702 * mu_diff / c))
            outcome_entropy = -win_prob * math.log(win_prob + 1e-10) - (1 - win_prob) * math.log(1 - win_prob + 1e-10)

            skill_similarity = 1.0 / (1.0 + abs(p1.mu - p2.mu))
            
            combined_score = outcome_entropy * skill_similarity
            
            if combined_score > best_score:
                best_score = combined_score
                best_pair = (i, j)
    
    return best_pair
    
def get_optimal_image_size(img, max_width=12, max_height=7):
    """Calculate optimal display size maintaining aspect ratio"""
    img_width, img_height = img.size
    aspect_ratio = img_width / img_height
    
    #Start with max dimensions and scale down as needed
    display_width = max_width
    display_height = max_height
    
    #Adjust based on aspect ratio
    if aspect_ratio > display_width / display_height:
        #Image is wider - constrain by width
        display_height = display_width / aspect_ratio
    else:
        #Image is taller - constrain by height  
        display_width = display_height * aspect_ratio
    
    return display_width, display_height

def compare(folder_path: str, img1_name: str, img2_name: str, match_count: int):
    print(f"'{img1_name}' vs '{img2_name}'")
    choice = None

    try:
        img1 = Image.open(os.path.join(folder_path, img1_name))
        img2 = Image.open(os.path.join(folder_path, img2_name))
    except Exception as e:
        print(f"Error opening images: {e}")
        choice = 'quit'
    
    #Find figure size based on source image sizes
    img1_width, img1_height = get_optimal_image_size(img1)
    img2_width, img2_height = get_optimal_image_size(img2)
    display_width = max(img1_width, img2_width)
    display_height = max(img1_height, img2_height)
    fig_width = display_width * 2 + 1
    fig_height = display_height + 1.5
    
    #Set up the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height))
    fig.suptitle(f"Press A/B to choose winner (ESC to quit). Match {match_count}", fontsize=14, fontweight='bold')
    ax1.imshow(img1)
    ax1.set_title(f"A: {os.path.basename(img1_name)}", fontsize=12, fontweight='bold', color='blue')
    ax1.axis('off')
    ax2.imshow(img2)
    ax2.set_title(f"B: {os.path.basename(img2_name)}", fontsize=12, fontweight='bold', color='red')
    ax2.axis('off')
    
    #Set up the window
    plt.tight_layout()
    try:
        mngr = plt.get_current_fig_manager()
        mngr.set_window_title("Image Tournament - Choose Winner")
        if hasattr(mngr, 'window'):
            mngr.window.wm_geometry("+100+100")
    except:
        pass  #Ignore if window manager operations fail
    
    def on_key(event):
        """Handle keyboard input"""
        nonlocal choice
        if event.key in ['a', 'A']:
            choice = 'A'
            plt.close(fig)
        elif event.key in ['b', 'B']:
            choice = 'B'
            plt.close(fig)
        elif event.key == 'enter':
            choice = 'default'
            plt.close(fig)
        elif event.key == 'escape':
            choice = 'quit'
            plt.close(fig)
    
    #Event Handler
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    plt.show()
    
    if choice == 'quit':
        return None
    elif choice == 'A':
        print(f"Winner: {img1_name}")
        return (1, 0)
    elif choice == 'B':
        print(f"Winner: {img2_name}")
        return (0, 1)
    else:
        # Shouldn't happen, but fallback
        return (1, 0)

def tournament(folder_path: str, images: list[str], init_scores: list[int]) -> list[int]:
    model = ThurstoneMostellerPart(tau=0, beta=BETA)
    max_matches = len(images) * int(math.log2(len(images)) * 3)
    
    players = [model.rating(mu=score) for score in init_scores]
    seen_matches = set()
    player_matchcount = [0 for player in players]
    match_count = 0
    while True:
        best_pair = select_next_match(players, seen_matches, player_matchcount, PLAYER_MATCH_LIMIT)
        idx1, idx2 = 0, 0
        if best_pair:
            idx1, idx2 = best_pair
            player_matchcount[idx1] += 1
            player_matchcount[idx2] += 1
        else:
            print(f"Ran out of allowed matches after {match_count} matches.")
            break
        seen_matches.add((min(idx1, idx2), max(idx1, idx2)))

        score1, score2 = 0, 0
        try:
            score1, score2 = compare(folder_path, images[idx1], images[idx2], match_count)
        except:
            print(f"Exited early after {match_count} matches.")

        #print(f"Conducting match between {idx1} and {idx2}")
        update_match(model, players, idx1, idx2, score1, score2)

        match_count += 1

        if match_count > max_matches:  #Safety limit
            print(f"Reached match limit without convergence. {match_count} matches conducted.")
            break
        
    print(f"Final average sigma: {sum(p.sigma for p in players) / len(players)}")

    final_scores = [player.mu for player in players]
    return final_scores

def bucketing(folder_path: str, num_buckets: int, images: list[str], scores: list[float]):
    if num_buckets <= 0:
        raise ValueError("num_buckets must be greater than 0")

    #Setup
    ranked = sorted(zip(images, scores), key=lambda x: x[1], reverse=True)
    total_images = len(images)
    bucket_size = max(1, total_images // num_buckets)

    #Make Bucket Folders
    bucket_dirs = []
    for i in range(0, num_buckets):
        bucket_dir = os.path.join(folder_path, f"bucket_{i}")
        os.makedirs(bucket_dir, exist_ok=True)
        bucket_dirs.append(bucket_dir)

    #Bucketing
    for i, (image, score) in enumerate(ranked):
        bucket_index = min(i // bucket_size, num_buckets - 1)
        source_path = os.path.join(folder_path, image)
        dest_path = os.path.join(folder_path, f"bucket_{bucket_index}", image)
        shutil.move(source_path, dest_path)

    #Record Data
    score_file_path = os.path.join(folder_path, "scores.txt")
    with open(score_file_path, "w", encoding="utf-8") as f:
        for image, score in ranked:
            f.write(f"{image}\t{score}\n")
    print(f"\nImages distributed into {num_buckets} buckets:")

def main():
    #Setup
    folder_path: str = ""
    while True:
        folder_path = input("Enter path to image folder: ").strip().strip('"\'')
        if not os.path.exists(folder_path):
            print(f"Folder path does not exist: {folder_path}")
        else:
            break
    images = load_images_from_folder(folder_path)
    print(f"Folder found, {len(images)} images found.")
    num_buckets = int(input("How many buckets? "))

    #Tournament
    init_scores = generate_starting_scores(folder_path, images)
    final_scores = tournament(folder_path, images, init_scores)

    #Wrapup
    bucketing(folder_path, num_buckets, images, final_scores)
    print("Tournament Completed")

if __name__ == "__main__":
    main()
