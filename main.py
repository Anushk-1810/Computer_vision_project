import streamlit as st
import time
import os

def main():
    # Streamlit App Title
    st.title("Football Video Analysis App")

    # Sidebar for Inputs
    st.sidebar.header("Upload Video")
    uploaded_file = st.sidebar.file_uploader("Choose a video", type=["mp4", "avi"])

    if uploaded_file is not None:
        st.sidebar.write("Uploaded file:", uploaded_file.name)

        # Step-by-Step Simulation
        steps = [
            "Reading the video...",
            "Tracking objects...",
            "Estimating camera movement...",
            "Transforming view...",
            "Interpolating ball positions...",
            "Estimating speed and distance...",
            "Assigning teams...",
            "Annotating output frames...",
            "Saving the processed video..."
        ]

        # Simulate each step
        for step in steps:
            st.write(f"**{step}**")
            with st.spinner(step):
                time.sleep(1)  # Simulate time delay for each step

        # Display Pre-Processed Output Video
        output_video_path = "output_videos/output_video.avi"
        if os.path.exists(output_video_path):
            st.write("**Step Completed: Displaying the processed video**")
            st.video(output_video_path)

            # Provide Download Option
            st.success("Video processing complete! Download the output video below:")
            with open(output_video_path, "rb") as video_file:
                st.download_button(
                    label="Download Processed Video",
                    data=video_file,
                    file_name="processed_video.avi",
                    mime="video/x-msvideo"
                )
        else:
            st.error("Output video not found in the folder. Please ensure it exists in 'output_videos/output_video.avi'.")

if __name__ == '__main__':
    main()
