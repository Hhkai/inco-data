for i in ./*; do
	if [ -d "$i" ]; then 
		echo "$i is a direcotry"
	elif [ -f "$i" ]; then
		echo "$i is a file"
		echo "aaa"
	fi
done
