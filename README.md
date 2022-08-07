# Seam Carving
Seam carving project in Computer Graphics course at Tel Aviv University.

The program does image resizing using the seam carving algorithm and other basic image processing operations.

The program is a command-line application with the following options:

**--image_path** (str)– An absolute/relative path to the image you want to process

**--output_dir** (str)– The output directory where you will save your outputs

**--height** (int) – the output image height

**--width** (int) – the output image width

**--resize_method** (str) – a string representing the resize method. Could be one of the following: [‘nearest_neighbor’,
‘seam_carving’]

**--use_forward_implementation** – a boolean flag indicates if forward looking
energy function is used or not.

**--output_prefix** (str) – an optional string which will be used as a prefix to the
output files. If set, the output files names will start with the given prefix. For seam carving, we will output two images, the resized image, and visualization of the chosen seems. So if --output_prefix is set to “my_prefix” then the output will be my_prefix_resized.png and my_prefix_horizontal _seams.png, my_prefix_vertical_seams.png. If the prefix is not set, then we will chose “img” as a default prefix.

