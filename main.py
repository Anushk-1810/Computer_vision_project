# from utils import read_video, save_video
# from trackers import Tracker
# import cv2
# import numpy as np
# from team_assigner import TeamAssigner
# from player_ball_assigner import PlayerBallAssigner
# from camera_movement_estimator import CameraMovementEstimator
# from view_transformer import ViewTransformer
# from speed_and_distance_estimator import SpeedAndDistance_Estimator


# def main():
#     # Read Video
#     video_frames = read_video('input_videos/08fd33_4.mp4')

#     # Initialize Tracker
#     tracker = Tracker('models/best.pt')

#     tracks = tracker.get_object_tracks(video_frames,
#                                        read_from_stub=True,
#                                        stub_path='stubs/track_stubs.pkl')
#     # Get object positions 
#     tracker.add_position_to_tracks(tracks)

#     # camera movement estimator
#     camera_movement_estimator = CameraMovementEstimator(video_frames[0])
#     camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
#                                                                                 read_from_stub=True,
#                                                                                 stub_path='stubs/camera_movement_stub.pkl')
#     camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)


#     # View Trasnformer
#     view_transformer = ViewTransformer()
#     view_transformer.add_transformed_position_to_tracks(tracks)

#     # Interpolate Ball Positions
#     tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

#     # Speed and distance estimator
#     speed_and_distance_estimator = SpeedAndDistance_Estimator()
#     speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

#     # Assign Player Teams
#     team_assigner = TeamAssigner()
#     team_assigner.assign_team_color(video_frames[0], 
#                                     tracks['players'][0])
    
#     for frame_num, player_track in enumerate(tracks['players']):
#         for player_id, track in player_track.items():
#             team = team_assigner.get_player_team(video_frames[frame_num],   
#                                                  track['bbox'],
#                                                  player_id)
#             tracks['players'][frame_num][player_id]['team'] = team 
#             tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    
#     # Assign Ball Aquisition
#     player_assigner =PlayerBallAssigner()
#     team_ball_control= []
#     for frame_num, player_track in enumerate(tracks['players']):
#         ball_bbox = tracks['ball'][frame_num][1]['bbox']
#         assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

#         if assigned_player != -1:
#             tracks['players'][frame_num][assigned_player]['has_ball'] = True
#             team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
#         else:
#             team_ball_control.append(team_ball_control[-1])
#     team_ball_control= np.array(team_ball_control)


#     # Draw output 
#     ## Draw object Tracks
#     output_video_frames = tracker.draw_annotations(video_frames, tracks,team_ball_control)

#     ## Draw Camera movement
#     output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)

#     ## Draw Speed and Distance
#     speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)

#     for frame in output_video_frames:
#         cv2.imshow('Output Video', frame)
#         if cv2.waitKey(30) & 0xFF == ord('q'):  # Press 'q' to exit early
#             break

#     cv2.destroyAllWindows()
#     # Save video
#     save_video(output_video_frames, 'output_videos/output_video.avi')

# if __name__ == '__main__':
#     main()

import streamlit as st

import cv2
import numpy as np
import tempfile
import os
from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
import os
import gdown


def download_model():
    model_path = 'models/best.pt'
    # Ensure the directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    url = 'YOUR_MODEL_URL'
    gdown.download(url, model_path, quiet=False)


# Call this function before loading the model
download_model()



# Updated read_video function
def read_video(file):
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file.write(file.read())  # Write the uploaded file content to temp file
    temp_file.close()

    # Read video frames using OpenCV
    cap = cv2.VideoCapture(temp_file.name)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    os.unlink(temp_file.name)  # Delete the temporary file
    return frames


def main():
    # Streamlit App Title
    st.title("Football Video Analysis App")

    # Sidebar for Inputs
    st.sidebar.header("Upload Video")
    uploaded_file = st.sidebar.file_uploader("Choose a video", type=["mp4", "avi"])

    if uploaded_file is not None:
        st.sidebar.write("Uploaded file:", uploaded_file.name)

        # Process Video
        st.write("Processing video... This might take some time.")
        video_frames = read_video(uploaded_file)

        # Initialize Tracker
        tracker = Tracker('models/best.pt')

        # Object Tracking
        st.write("Tracking objects...")
        tracks = tracker.get_object_tracks(video_frames,
                                           read_from_stub=True,
                                           stub_path='stubs/track_stubs.pkl')
        tracker.add_position_to_tracks(tracks)

        # Camera Movement Estimation
        st.write("Estimating camera movement...")
        camera_movement_estimator = CameraMovementEstimator(video_frames[0])
        camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
            video_frames,
            read_from_stub=True,
            stub_path='stubs/camera_movement_stub.pkl'
        )
        camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

        # View Transformation
        st.write("Transforming view...")
        view_transformer = ViewTransformer()
        view_transformer.add_transformed_position_to_tracks(tracks)

        # Interpolate Ball Positions
        tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

        # Speed and Distance Estimation
        st.write("Estimating speed and distance...")
        speed_and_distance_estimator = SpeedAndDistance_Estimator()
        speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

        # Team Assignment
        st.write("Assigning teams...")
        team_assigner = TeamAssigner()
        team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

        for frame_num, player_track in enumerate(tracks['players']):
            for player_id, track in player_track.items():
                team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
                tracks['players'][frame_num][player_id]['team'] = team
                tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

        # Ball Assignment to Players
        st.write("Assigning ball to players...")
        player_assigner = PlayerBallAssigner()
        team_ball_control = []
        for frame_num, player_track in enumerate(tracks['players']):
            ball_bbox = tracks['ball'][frame_num][1]['bbox']
            assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

            if assigned_player != -1:
                tracks['players'][frame_num][assigned_player]['has_ball'] = True
                team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
            else:
                team_ball_control.append(team_ball_control[-1])
        team_ball_control = np.array(team_ball_control)

        # Draw Output
        st.write("Annotating output frames...")
        output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
        output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
        speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

        # Save Processed Video
        st.write("Saving the processed video...")
        output_path = 'output_videos/output_video.avi'
        save_video(output_video_frames, output_path)

        # Display Processed Video
        st.video(output_path)

        st.success("Video processing complete! Download the output video below:")
        with open(output_path, "rb") as video_file:
            st.download_button("Download Video", video_file, file_name="processed_video.avi")


if __name__ == '__main__':
    main()
