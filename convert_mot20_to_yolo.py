import os
import cv2

initial_path = '/mnt/d/Downloads/MOT20/'
# Path to MOT20
mot_path = initial_path + 'MOT20/train'

# Path to output labels in YOLO format
output_labels = 'labels'

for seq in os.listdir(mot_path):
    gt_path = os.path.join(mot_path, seq, 'gt', 'gt.txt')
    img_path = os.path.join(mot_path, seq, 'img1')

    print(f'Processing {seq}...')

    # Read images
    sample_img = cv2.imread(os.path.join(img_path, '000001.jpg'))
    H, W, _ = sample_img.shape

    # Create output folder
    out_dir = os.path.join(output_labels, seq)
    os.makedirs(out_dir, exist_ok=True)

    with open(gt_path, 'r') as f:
        for line in f:
            tokens = line.strip().split(',')
            frame_id = int(tokens[0])
            target_id = int(tokens[1])
            x = float(tokens[2])
            y = float(tokens[3])
            w = float(tokens[4])
            h = float(tokens[5])
            conf = float(tokens[6])
            cls_id = int(tokens[7])

            # class_id = 1 for humans
            if cls_id != 1:
                continue

            # Normalize
            x_center = (x + w / 2) / W
            y_center = (y + h / 2) / H
            w_norm = w / W
            h_norm = h / H

            # Name file
            out_filename = f"{frame_id:06d}.txt"
            out_path = os.path.join(out_dir, out_filename)

            # Write to file
            with open(out_path, 'a') as out_file:
                out_file.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

print(f'Successully converted MOT20 --> YOLO format! | {out_path}')
