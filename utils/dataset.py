from PIL import Image
import torch
from torchvision import transforms
import os


def add_overlap(vectors, overlap_ratio, epsilon=0.0):
    n = len(vectors)
    m = vectors[0].size(0)

    # Verify all vectors are of the same length
    for vector in vectors:
        if vector.size(0) != m:
            raise ValueError("All vectors must have the same length")

    # Calculate the overlap in terms of number of elements
    overlap = int(overlap_ratio / 100 * m)

    # Calculate the shift for each vector (number of leading and trailing epsilons)
    shift = m - overlap

    # Calculate the total length of each vector including epsilon padding
    total_length = m + shift * (n - 1)

    # Initialize a tensor to store the shifted vectors, filled with epsilon
    shifted_vectors = torch.full((n, total_length), epsilon)

    # Generate each shifted vector
    for i in range(n):
        # Determine the start index for placing the elements of the current vector
        start_idx = i * shift

        # Place the elements of the current vector starting from the calculated index
        shifted_vectors[i, start_idx:start_idx + m] = vectors[i]

    return shifted_vectors


def read_and_flatten_image(file_path, resize_width, resize_height):
    # Define the image transformations
    transform = transforms.Compose([
        transforms.Resize((resize_width, resize_height)),  # Resize the image
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.ToTensor(),  # Convert to a tensor
    ])

    # Open the image file
    image = Image.open(file_path)

    # Apply the transformations
    image_tensor = transform(image)

    # Flatten the tensor
    flat_image_tensor = image_tensor.view(-1)

    return flat_image_tensor


def load_data(directory, resize_width, resize_height):
    # List all files in the directory
    files = [os.path.join(directory, file) for file in os.listdir(directory) if
             file.endswith(('.png', '.jpg', '.jpeg'))]

    # Initialize a list to store the image tensors
    image_tensors = []

    for file in files:
        # Use the function to read and flatten the image
        flat_image_tensor = read_and_flatten_image(file, resize_width, resize_height)

        # Append the flattened tensor to the list
        image_tensors.append(flat_image_tensor)

    # Stack all tensors into a single tensor
    final_tensor = torch.stack(image_tensors)

    return final_tensor


if __name__ == "__main__":
    # Example usage
    # Path to dataset directory
    directory = '../dataset'

    # Image dimensions
    resize_width = 128
    resize_height = 128

    # Process multiple images
    images_array = load_data(directory, resize_width, resize_height)
    print(images_array.shape)

    # Path to a specific image file
    file_path = '../dataset/pattern1.png'

    # Process a single image
    single_image_array = read_and_flatten_image(file_path, resize_width, resize_height)
    print(single_image_array.shape)
