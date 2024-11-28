import streamlit as st
import tempfile
from ultralytics import YOLO, SAM
import os
import pandas as pd
from openai import AzureOpenAI
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Cost estimation dictionary for dent, scratch, glass break, and accidents
DEFECT_COSTS = {
    "Dent": 500,  # Example cost for dent
    "scratch": 150,  # Example cost for minor scratch
    "Glass-Break": 800,  # Example cost for glass break
    "Accident": 2000,  # Example cost for accident repair
}

class AIAssistant:
    def __init__(self):
        """
        Initialize Azure OpenAI Client with robust configuration checking
        """
        # Retrieve credentials from environment variables
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_KEY")
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")

        # Validate credentials
        self.client = self.validate_credentials()

    def validate_credentials(self):
        """
        Validate Azure OpenAI credentials
        """
        # Check if credentials are present
        if not all([self.endpoint, self.api_key, self.deployment_name]):
            st.sidebar.error("""
            ❌ Azure OpenAI Configuration Incomplete
            """)
            return None

        try:
            # Attempt to create Azure OpenAI client
            client = AzureOpenAI(
                azure_endpoint=self.endpoint,
                api_key=self.api_key,
                api_version="2024-02-01"
            )
            return client
        except Exception as e:
            st.sidebar.error(f"Azure OpenAI Connection Error: {e}")
            return None

    def gen_ai(self, detection_data, total_estimated_cost):
        """
        Generate AI-powered insights about car damages
        """
        # Check if client is configured
        if not self.client:
            return {
                "error": "AI service not configured. Please check your Azure OpenAI credentials.",
                "details": {
                    "endpoint": bool(self.endpoint),
                    "api_key": bool(self.api_key),
                    "deployment_name": bool(self.deployment_name)
                }
            }

        try:
            # Enhanced prompt for generating insights
            system_prompt = """
            You are an expert automotive damage assessment specialist. 
            Given the detection details and the total estimated repair costs, provide a comprehensive report on the car's condition and repair strategy. 
            
            The report should include:
            1. Detailed description of each defect type, its severity, and cost estimate.
            2. Long-term implications of the defects.
            3. Recommended repair strategies for each defect.
            4. Preventive measures to avoid similar damages in the future.
            5. The impact of the defects on the vehicle’s resale value.
            6. Suggestions for minimizing the overall repair cost.

            The input data for the analysis includes:
            - Detection details with cost estimations.
            - Total estimated repair cost.
            """

            user_prompt = json.dumps({
                "detection_details": detection_data,
                "total_estimated_cost": total_estimated_cost
            })

            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user_prompt}],
                max_tokens=200
            )

            insights = response.choices[0].message.content
            return {"insights": insights}

        except Exception as e:
            return {
                "error": "AI insight generation failed",
                "details": str(e)
            }

def load_models(yolo_model_path, sam_model_path):
    """
    Load YOLO and SAM models with error handling.
    """
    try:
        yolo_model = YOLO(yolo_model_path)
        sam_model = SAM(sam_model_path)
        return yolo_model, sam_model
    except Exception as e:
        st.sidebar.error(f"Error loading models: {e}")
        return None, None

def process_image(uploaded_file, yolo_model, sam_model):
    """
    Process the uploaded image with YOLO detection and SAM segmentation.
    """
    # Save the uploaded image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        # YOLO Detection
        yolo_results = yolo_model(tmp_file_path, save=True)

        # Prepare results for display
        detection_data = []
        total_estimated_cost = 0

        for result in yolo_results:
            for box in result.boxes:
                confidence = box.conf[0]
                class_name = yolo_model.names[int(box.cls[0])]

                # Assign cost based on class and confidence
                cost = None
                if class_name == "Dent":
                    cost = DEFECT_COSTS["Dent"]
                elif class_name == "Scratch":
                    cost = DEFECT_COSTS["scratch"]
                elif class_name == "Glass-Break":
                    cost = DEFECT_COSTS["Glass-Break"]
                elif class_name == "Accident":
                    cost = DEFECT_COSTS["Accident"]

                if cost is not None:
                    total_estimated_cost += cost
                    detection_data.append({
                        "Defect Type": class_name,
                        "Severity": "Major" if confidence > 0.4 else "Minor",
                        "Estimated Cost": f"${cost}"
                    })

        # SAM Segmentation
        segmentation_images = []
        for result in yolo_results:
            boxes = result.boxes.xyxy
            if len(boxes) > 0:
                sam_results = sam_model(result.orig_img, bboxes=boxes, verbose=False, save=True, device="cpu")
                for idx, seg_result in enumerate(sam_results):
                    temp_seg_image_path = f"segmentation_result_{idx}.jpg"
                    seg_result.save(temp_seg_image_path)
                    segmentation_images.append(temp_seg_image_path)

        # Clean up temporary file
        os.unlink(tmp_file_path)

        return detection_data, total_estimated_cost, segmentation_images

    except Exception as e:
        st.error(f"Error processing image: {e}")
        os.unlink(tmp_file_path)
        return [], 0, []

def main():
    # Page configuration
    st.set_page_config(
        page_title="Car Defect Detection System", 
        page_icon=":car:", 
        layout="wide"
    )

    # Initialize AI Assistant
    ai_assistant = AIAssistant()

    # Title and description
    st.title("CarScanAI")
    st.markdown("""
    ### Detect and Estimate Repair Costs for Car Damages
    Upload an image of a car, and our AI will:
    - Detect defects like dents, scratches, glass breaks, and accidents
    - Estimate repair costs
    - Generate segmentation masks
    - Provide AI-powered damage insights
    """)

    # Sidebar configuration
    st.sidebar.header("🛠️ System Configuration")

    # Model paths
    yolo_model_path = "auto_defect.pt"
    sam_model_path = "sam2_b.pt"

    # Load models
    st.sidebar.subheader("Model Status")
    yolo_model, sam_model = load_models(yolo_model_path, sam_model_path)

    if yolo_model is None or sam_model is None:
        st.sidebar.error("Failed to load models. Please check model files.")
        return

    # File uploader
    uploaded_file = st.file_uploader(
        "📤 Upload Car Image", 
        type=["jpg", "jpeg", "png"],
        help="Upload an image to analyze car defects"
    )

    if uploaded_file is not None:
        # Create columns for layout
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📸 Uploaded Image")
            st.image(uploaded_file, use_container_width=True)

        with col2:
            st.subheader("📊 Damage Analysis")

            # Process image
            detection_data, total_estimated_cost, segmentation_images = process_image(uploaded_file, yolo_model, sam_model)

            if detection_data:
                # Display detection results as a DataFrame
                df = pd.DataFrame(detection_data)
                st.dataframe(df, hide_index=True)

                # Total cost estimation
                st.metric("Total Estimated Repair Cost", f"${total_estimated_cost}")
            else:
                st.info("No defects detected in the image.")

        # Segmentation results
        if segmentation_images:
            st.subheader("🔍 Segmentation Masks")
            seg_cols = st.columns(len(segmentation_images))
            for i, img_path in enumerate(segmentation_images):
                with seg_cols[i]:
                    st.image(img_path, caption=f"Segmentation Mask {i+1}", use_container_width=True)

        # AI-Powered Damage Insights
        if detection_data:
            st.subheader("🤖 AI Damage Insights")
            with st.spinner("Generating intelligent insights..."):
                ai_insights = ai_assistant.gen_ai(detection_data, total_estimated_cost)

                if "insights" in ai_insights:
                    st.info(ai_insights["insights"])
                elif "error" in ai_insights:
                    st.warning(ai_insights["error"])

                    if "details" in ai_insights:
                        st.json(ai_insights["details"])

if __name__ == "__main__":
    main()
