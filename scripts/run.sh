video_path='/media/anlab/4tb/projects/data_moving/video1_2.mp4'
outpath_save='/media/anlab/4tb/projects/data_moving/results_carton/'

cd ./tool   
# tracking with Bytetrack
python tracker_GDino_Bytetrack.py -i=$video_path -o=$outpath_save

# preprocess output of tracking
python draw_data_track_step1.py --i=$video_path --o=$outpath_save
