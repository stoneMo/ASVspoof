# download LA dataset 
# https://drive.google.com/file/d/1UGs1o2mDiBO9_iaN-0FupS8x0Tkb4xmt/view?usp=sharing
# download agot data from Googledrive

fileid="1UGs1o2mDiBO9_iaN-0FupS8x0Tkb4xmt"
filename="LA.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
