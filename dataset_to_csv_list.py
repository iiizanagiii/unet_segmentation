import os
import csv
import yaml 

def create_csv_file(images_folder, masks_folder, csv_file):
    # Get the list of image and mask filenames
    image_files = os.listdir(images_folder)
    mask_files = os.listdir(masks_folder)

    # Make sure the number of image and mask files match
    if len(image_files) != len(mask_files):
        print("Number of image files does not match the number of mask files.")
        exit()

    # Create a list of dictionaries for each image and mask pair
    data = []
    for image_file, mask_file in zip(image_files, mask_files):
        data.append({'images': image_file, 'masks': mask_file})

    # Write the data to the CSV file
    with open(csv_file, 'w', newline='') as csvfile:
        fieldnames = ['masks', 'images']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for entry in data:
            writer.writerow(entry)

    print(f"CSV file '{csv_file}' created successfully.")

if __name__ == "__main__":
    with open('./config.yaml') as config_file:
        config =  yaml.safe_load(config_file)

    # Define the paths to your folders
    images_folder = config['augment_image_dir'] #'./data_augment/orange-tabs/images'
    masks_folder =  config['augment_mask_dir'] #'./data_augment/orange-tabs/mask'

    # Define the CSV file path
    csv_file = config['dataset_csv_file'] # './data_augment/orange-tabs/image_mask_pairs.csv'
    create_csv_file(images_folder, masks_folder, csv_file)