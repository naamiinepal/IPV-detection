#! /bin/bash

# Set the text color to red
tput setaf 1

if (($# != 1)); then
    echo 'Error: Please pass the path containing the config files in the first argument'
    exit 1
fi

config_folder=$1

if [ ! -d $config_folder ]; then
    echo "Error: $config_folder is not a valid directory"
    exit 1
fi

# Iterate over the config files in the specified directory
for config_file in $config_folder/fold*.yaml; do
    # Normalize the path to remove double slashes
    config_file=$(echo $config_file | sed -e 's|//|/|g')

    # Set the text color to yellow
    tput setaf 3

    echo "Performing kfold for: $config_file"

    # Reset the text color to the default
    tput sgr0

    # Run the k fold validation using the current config file
    ./trainer.py fit --config $config_file
done
