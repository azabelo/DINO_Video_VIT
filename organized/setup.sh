#!/bin/bash

#download dataset

# Specify the URL of the file to download
file_url="http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar"

# Specify the desired name for the downloaded file
file_name="hmdb51_dataset.rar"

# Download the file using curl
curl -o "$file_name" "$file_url"
# List all files in the current directory
files=$(ls -p)

# Print the list of files
echo "Files in the current directory:"
echo "$files"


# Install unrar if not already installed
if ! command -v unrar &> /dev/null; then
    echo "Installing unrar..."
    sudo apt-get install unrar
fi

# Create a directory to extract rar files
mkdir -p hmdb51_unrared

#unzip big file
unrar x "$file_name" hmdb51_unrared


cd hmdb51_unrared

# Get the current directory
current_directory=$(pwd)

# List all files in the current directory
files=$(ls -p "$current_directory" | grep -v /)

# Print the list of files
for file in $files; do
    echo "$file"
done

# Extract .rar files
rar_files=$(ls -p "$current_directory" | grep -v / | grep -e "\.rar$")
for rar_file in $rar_files; do
    unrar x "$rar_file"
done

## Delete all .rar files in the current directory
#rm *.rar

cd ..

# Get the current working directory
current_dir=$(pwd)

# Print the current working directory
echo "Current working directory: $current_dir"


# Specify the path to the requirements.txt file
requirements_file="organized/requirements.txt"

# Check if requirements.txt file exists
if [ ! -f "$requirements_file" ]; then
    echo "Error: requirements.txt file not found."
    exit 1
fi

# Read each line in requirements.txt and install the packages
while read -r package; do
    # Ignore comments and empty lines
    if [[ $package != \#* ]] && [ -n "$package" ]; then
        echo "Installing package: $package"
        pip install "$package"
    fi
done < "$requirements_file"

cd ..
#run experiment
python3 organized/pipeline.py current_directory