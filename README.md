
# Sports similarity (attempt siamese)
Project aims at finding similarity between football players (same team as well). 

# Dataset creation
2 videos of football matches are being used, (many jump cuts have been made in order to get maximum field view)
YOLOv5( crowd dataset) + byteTrack has been used to extract each player in the respective folders, these extracting of  players are based on unique ids. After images are collected these then are manually checked for chosing anchors and negatives. Positives (in case we need triplet loss) is selected from anchor images. 
For creating final training dataset, an image array records (anchor , positive and anchor , negative) alternately, alongwith saving labels (0 for similar pairs, and 1 otherwise).


# Network
The embedding network looks like this. 

![alt text](https://github.com/manish33scss/sports_similarity_siamese/blob/main/meesi_siam_.png)

Followed by, distance layer (Euclidean distance) and for loss, contrastive loss has been used.

#usage
```bash
Run : 
sports_similarity_siamese.py
```
Load the directory where images are stored in directories ex : person1, person2, ... 
