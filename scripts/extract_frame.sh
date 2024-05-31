#!/bin/bash
if [ ! -d "/Ego4d/Ego4D_LookAtMe/video_imgs"  ];then
	mkdir "/Ego4d/Ego4D_LookAtMe/video_imgs"
fi
for file in `ls /Ego4D_LookAtMe/videos/*`
do
	name=$(basename $file .mp4)
	echo "$name"
	PTHH=/Ego4d/Ego4D_LookAtMe/video_imgs/$name
	if [ ! -d "$PTHH"  ];then
		mkdir "$PTHH"
	fi
	ffmpeg -i "$file" -f image2 -vf fps=30 -qscale:v 2 "$PTHH/img_%05d.jpg"
done
