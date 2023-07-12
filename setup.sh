#!/bin/bash

#download dataset

# Specify the URL of the file to download
curl https://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar --output hmdb51_dataset.rar

# Install unrar if not already installed
if ! command -v unrar &> /dev/null; then
    echo "Installing unrar..."
    sudo apt-get install unrar
fi


#unzip big file
mkdir hmdb51_unrared
mv hmdb51_dataset.rar hmdb51_unrared/hmdb51_dataset.rar
cd hmdb51_unrared
unrar x hmdb51_dataset.rar

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
rm *.rar

cd ..

# Get the current working directory
current_dir=$(pwd)

# Print the current working directory
echo "Current working directory: $current_dir"

# Specify the path to the requirements.txt file
requirements_file="requirements.txt"

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
cd DINO_Video_VIT
#run experiment
python3 pipeline.py