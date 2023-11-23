name=$1
output_dir=$2

echo "name: $name"
echo "output directory: $output_dir"

cd $output_dir

gcc -o $name main_axis0.c $name.c

./$name