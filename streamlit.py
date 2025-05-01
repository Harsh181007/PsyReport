import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import numpy as np
from streamlit_drawable_canvas import st_canvas
from streamlit_option_menu import option_menu
import os
from collections import Counter
from model_loaders.house_loader import predict
from model_loaders.color_loader import color_predict
from transformers import pipeline
from fpdf import FPDF
import io
import unicodedata



# Sidebar
with st.sidebar:
    st.image('images/logo.png', width=250)
    
    # Bootstrap icons
    selected = option_menu(
        'PsyReport',
        ['Overview', 'House Drawing Test', 'Tree Drawing Test', 'Person Drawing Test','Color Drawing Test'],
        icons=['house', 'house-door-fill', 'tree-fill', 'person-arms-up', 'square-half'],
        default_index=0
    )



# Overview Page
if selected == "Overview":    
    st.markdown(
        "<h1 style='text-align: center;'>PsyReport: AI-Powered Psychological Insights from Children's Drawings</h1>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<h4 style='text-align: center; font-style: italic;'>🖍️ “Sometimes, a child's drawing speaks louder than words.”</h4>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ## 🌟 Why PsyReport?  
        Children's drawings are more than just art—they reveal emotions, thoughts, and subconscious feelings.  
        
        Globally, around **8% of children and 15% of adolescents (ages 3-14)** experience mental disorders such as anxiety, depression, and behavioral issues.
        
        **PsyReport** is an **AI-powered psychological analysis tool** that interprets **House, Tree, and Person (HTP) drawings** to uncover a child's **emotional state, personality traits, and mental well-being**.  
        It helps parents, teachers, and psychologists gain valuable insights using **scientific methods** for early emotional and psychological assessment.  


        ## 🎯 What Makes PsyReport Unique?  
        - **🔍 Hidden Emotion Detection** – Reveals emotions such as joy, fear, stress, or insecurity from subtle drawing patterns.  
        - **🧑‍⚕️ Early Mental Health Support** – Identifies signs of emotional distress **before they manifest in behavior**.  
        - **🖌️ Personality & Social Traits** – Determines whether a child is **introverted, extroverted, independent, or emotionally reserved**.  
        - **📜 AI-Powered Reports** – Uses advanced **deep learning and NLP** models to generate **detailed psychological assessments**.  
        - **🚀 Instant & Research-Backed** – Based on the widely recognized **HTP (House-Tree-Person) Test**, a **validated tool in child psychology**.  

        ## 🏠🌳🧍‍♂️ The Power of the HTP Test  
        The **House-Tree-Person (HTP) Test** is a **globally used projective psychological test** designed to **uncover subconscious emotions and cognitive patterns** through drawings.  
        - **🏠 House:** Represents a child’s perception of their home environment and relationships.  
        - **🌳 Tree:** Symbolizes emotional strength, personal growth, and self-image.  
        - **🧍 Person:** Reflects self-perception, social confidence, and identity.  

        By analyzing these drawings, PsyReport provides **personalized AI-driven interpretations** that offer **actionable insights for parents and psychologists**.  

        ## 🚀 How to Use PsyReport?  
        1. **Select a Drawing Type:** Choose **House, Tree, or Person (HTP) analysis**.  
        2. **Upload or Draw:** Upload an image of a drawing or use our **interactive canvas**.  
        3. **AI Interpretation:** Our advanced AI model analyzes the sketch for psychological patterns.  
        4. **Receive a Comprehensive Report:** Get a **personalized psychological assessment** with expert-backed insights.  

        ## 📜 What’s Included in Your AI Report?  
        - **Emotional State Analysis** – Detects stress, anxiety, happiness, or emotional distress.  
        - **Personality Traits** – Identifies introversion, extroversion, self-confidence, or withdrawal.  
        - **Psychological Well-Being** – Assesses cognitive and emotional balance.  
        - **Symbolic Interpretations** – Highlights how **specific drawing features** relate to emotions.  
        - **Guidance & Recommendations** – Provides **practical steps for parents, educators, or therapists**.  

        ### 🔬 Backed by Science & AI  
        - **HTP Test (House-Tree-Person):** A widely used child psychology tool in schools and therapy.  
        - **Research on Child Development:** Studies confirm that **drawings can reflect underlying psychological states**.  
        - **AI & Mental Health:** Deep learning models significantly improve the **accuracy of psychological assessments**.  

        ## 🌍 Why PsyReport Matters?  
        - **🚀 Early Detection = Better Outcomes** – Identifying emotional issues early leads to better **mental and emotional development**.  
        - **🎨 Expressive Freedom** – Many children **express their emotions better through drawings** than words.  
        - **📊 AI for Psychology** – Our technology makes **psychological insights accessible, accurate, and actionable** for everyone.  

        **PsyReport bridges the gap between Psychology and AI, offering a scientific yet compassionate way to understand children's emotions.**  

        """
    )



# Canvas Image Saving Function
def save_canvas_image(image_array):
    """
    Convert the canvas drawing to an RGB image with a white background and black strokes.
    """
    # Convert NumPy array to PIL image (RGBA mode)
    image = Image.fromarray((image_array * 255).astype(np.uint8))  
    image = image.convert("L")
    image = ImageOps.invert(image)
    image = image.convert("RGB")  

    # Save final image
    image.save(IMAGE_PATH)
    return IMAGE_PATH

# NLP models pipeline
@st.cache_resource
def load_nlp_models():
    return {
        "nlp_ner": pipeline("ner", model="dslim/bert-base-NER"),
        "summarizer": pipeline("summarization", model="facebook/bart-large-cnn"),
        "paraphraser": pipeline("text2text-generation", model="t5-small")
    }

nlp_model = load_nlp_models()
nlp_ner = nlp_model["nlp_ner"]
summarizer = nlp_model["summarizer"]
paraphraser = nlp_model["paraphraser"]


# Generate a dynamic report using NLP pipeline
def generate_report(result):
    report = feature_based_reports.get(result, None)

    if report:
        feature_text = ", ".join(report["features"])  

        # Generate base report with original psychological meaning
        generated_text = report["base_report"].format(features=feature_text)

        # Apply subtle paraphrasing
        paraphrased_report = paraphraser(generated_text, max_length=120, do_sample=True)
        paraphrased_text = paraphrased_report[0]['generated_text']

        # Ensure summarization does not remove key reasoning
        max_len = min(120, int(len(paraphrased_text) * 0.9))  
        summarized_report = summarizer(paraphrased_text, max_length=max_len, min_length=60, do_sample=False)
        final_report = summarized_report[0]['summary_text']

        # Store in session_state
        st.session_state["final_report"] = final_report  

        # Display final report
        st.info(report["title"])
        st.write(final_report)  


def normalize_text(text):
    """Normalize text to remove special Unicode characters that might not be supported."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def generate_pdf(report_text, image_path):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(200, 10, "Psychological Analysis Report", ln=True, align="C")
    pdf.ln(10)

    try:
        img = Image.open(image_path)
        img_width, img_height = img.size

        max_width = 120  
        aspect_ratio = img_height / img_width
        scaled_height = max_width * aspect_ratio  

        # Calculate center position (A4 width is 210mm)
        x_center = (210 - max_width) / 2  # Center horizontally
        y_position = 40  # Fixed top position

        # Draw border around image
        border_padding = 5  # Padding for the border
        pdf.set_draw_color(0, 0, 0)  # Black border
        pdf.rect(x_center - border_padding, y_position - border_padding, max_width + 2 * border_padding, scaled_height + 2 * border_padding)

        # Add image
        pdf.image(image_path, x=x_center, y=y_position, w=max_width)

        # Move text below the image with spacing
        y_text = y_position + scaled_height + 20  # Add gap after image
        pdf.set_y(y_text)
    except Exception as e:
        print(f"Error loading image: {e}")

    # Normalize text to avoid encoding issues
    report_text = normalize_text(report_text)

    # Add report text
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, report_text)

    # Generate PDF as a byte string
    pdf_bytes = pdf.output(dest="S")  # No need to encode

    # Convert to BytesIO object
    pdf_buffer = io.BytesIO(pdf_bytes)

    return pdf_buffer



# House Drawing Analysis Page
if selected == "House Drawing Test":    
    st.title("🏠 House Drawing Test")
    
    st.markdown("""
        ## Introduction
        The **House Test** is a psychological assessment that analyzes a child's emotions and personality through their drawings. Our AI-powered system evaluates features like structure, proportions, and details to classify emotional traits. An explainable AI (XAI) module provides insights into the reasoning behind the results, helping parents better understand their child's psychological state.

        ## Important Instructions for the Drawing Analysis
        - **Stay Calm & Focused** – Ensure the child is in a normal, undisturbed mood before drawing. This is a serious psychological assessment, and emotional state can influence the outcome.  
        - **Create a 2D Drawing** – The house should be a simple two-dimensional sketch, not a 3D perspective.  
        - **Include Key Features** – The drawing can include elements such as doors, windows, chimneys, or any other details the child wants to add.  
        - **Shading is Allowed** – Any part of the drawing can be shaded if desired.  
        - **Use Only Black Ink** – The drawing should be made with black ink or a black digital pen for consistency in analysis.  
        """)


    # Directory where images will be stored
    PREDICT_PATH = "toPredict"
    os.makedirs(PREDICT_PATH, exist_ok=True)

    IMAGE_PATH = os.path.join(PREDICT_PATH, "predictHouse.png")

    # Layout for better organization
    col1, col2 = st.columns([1, 1])  

    with col1:  
        st.subheader("🖌️ Draw Your House Here")
        st.write("Use the canvas below to draw a house. Once finished, save it before predicting.")

        canvas_result = st_canvas(
            fill_color="rgb(255,255,255,1)",  
            stroke_width=3,
            stroke_color="black",
            background_color="white",
            height=224,
            width=224,
            drawing_mode="freedraw",
            key="canvas"
        )

        if st.button("💾 Save Drawing"):
            if canvas_result.image_data is not None:
                img_array = np.array(canvas_result.image_data)  # Convert canvas data to NumPy array

                if np.all(img_array[:, :, :3] == 255):  # Check if all pixels are white
                    st.warning("⚠️ Your canvas is empty! Please draw before saving.")
                else:
                    save_canvas_image(img_array)
                    st.success("✅ Drawing saved successfully!")

    with col2:  # Right side: Upload Image Section
        st.subheader("📤 Upload a House Drawing")
        st.write("If you've already drawn a house on paper, you can upload a scanned image here.")

        uploaded_file = st.file_uploader("Choose an image (PNG, JPG, JPEG)...", type=["png", "jpg", "jpeg"])


    # Prediction function using ENSEMBLE LEARNING (MAJORITY VOTING)
    def predict_image(image_path):
            """
            Pass the saved image to the models and return the majority prediction.
            """
            housePredict = [
                predict("model/house/house_model_10.tar", image_path),
                predict("model/house/house_model_12.tar", image_path),
                predict("model/house/house_model_15.tar", image_path),
            ]

            final_prediction = Counter(housePredict).most_common(1)[0][0]
            return final_prediction

    # Prediction Section
    st.subheader("🔍 Get Your Result")
    st.write("Once you have either drawn a house or uploaded an image, click the 'Predict' button below.")

    if st.button("🔍 Predict"):
        image_path = None

        if uploaded_file is not None:
            # Save uploaded image as the new input
            image = Image.open(uploaded_file)
            image.save(IMAGE_PATH)
            image_path = IMAGE_PATH
        elif os.path.exists(IMAGE_PATH):
            # If no upload, use the saved canvas image (if available)
            image_path = IMAGE_PATH

        # Process and predict
        if image_path:
            col1, col2 = st.columns([1, 2])  # Adjusting proportions for better readability

            with col1:  
                st.image(image_path, caption="🖼 Processed Image", width=210)  # Reduced size for a cleaner look

            with col2:  
                result = predict_image(image_path)

                # Define psychological features mapped to each outcome
                feature_based_reports = {
                        0: {
                            "title": "❌ **Prediction Result: Stress or Anxiety Detected**",
                            "traits": ["Stress", "Anxiety"],
                            "features": [
                                "Excessive amount of smoke",
                                "High details",
                                "Shaded roof"
                            ],
                            "base_report": (
                                "The drawing suggests signs of **stress, insecurity, or anxiety.** "
                                "Children experiencing these emotions might show {features}. "
                                "These details could indicate **inner tension, heightened sensitivity, or worry.** "
                                "Consider observing the child's **overall behavior and emotional well-being** to offer support."
                            ),
                        },
                        1: {
                            "title": "✅ **Prediction Result: Introverted & Thoughtful Personality**",
                            "traits": ["Low self-esteem", "Introverted", "Withdrawn"],
                            "features": [
                                "Absence of door",
                                "Absence of windows",
                                "Door located high above base line",
                                "Missing a chimney"
                            ],
                            "base_report": (
                                "The drawing indicates that the child might be **reserved, introspective, or cautious in social interactions.** "
                                "The presence of {features} suggests a **desire for isolation, withdrawal, or emotional distance.** "
                                "A missing door or windows may symbolize **difficulty expressing emotions or seeking connections.** "
                                "Encouraging **creative expression and open communication** can help such children feel more comfortable sharing their thoughts."
                            ),
                        },
                        2: {
                            "title": "🟡 **Prediction Result: Extroverted & Imaginative Personality**",
                            "traits": ["High self-esteem", "Extroverted", "Fantasizing"],
                            "features": [
                                "Very large door",
                                "Very large roof",
                                "Very large windows",
                                "Open door"
                            ],
                            "base_report": (
                                "The drawing reflects a **confident, socially open, and imaginative personality.** "
                                "Children who are extroverted and expressive may depict {features}, "
                                "suggesting a **welcoming and outgoing nature.** "
                                "A large door and windows may symbolize **openness to new experiences and a strong social inclination.** "
                                "Such children may be **adventurous, creative, and enthusiastic about interacting with others.**"
                            ),
                        }
                    }

                generate_report(result)

        else:
            st.warning("⚠️ Please draw an image and save it or upload a file before predicting.")
        

    # Check if a report exists before showing download button
    if "final_report" in st.session_state and st.session_state["final_report"]:
        pdf_buffer = generate_pdf(st.session_state["final_report"], IMAGE_PATH)

        # Provide download button
        st.download_button(
            label="Download Report",
            data=pdf_buffer,
            file_name="psychological_report.pdf",
            mime="application/pdf"
        )
    else:
        st.error("No report generated yet. Please generate a report first.")


# Tree Drawing Analysis Page
if selected == 'Tree Drawing Test':
    st.title("🌳 Tree Drawing Test")
    st.markdown("""
        ## Introduction  
        The **Tree Test** is a psychological assessment that evaluates a child's emotions and personality based on their tree drawing. The AI system analyzes features such as shape, structure, and details to classify emotional traits. Our explainable AI (XAI) module provides reasoning behind the results, helping parents understand their child's psychological state.  

        ## Important Instructions for the Drawing Analysis  
        - **Stay Calm & Focused** – Ensure the child is in a normal, undisturbed mood before drawing, as this is a serious psychological assessment.  
        - **Create a 2D Drawing** – The tree should be a simple two-dimensional sketch, not a 3D perspective.  
        - **Include Key Features** – The drawing can include roots, trunk, branches, leaves, or any other details.  
        - **Shading is Allowed** – Any part of the drawing can be shaded if desired.  
        - **Use Only Black Ink** – The drawing should be made with black ink or a black digital pen for consistency in analysis.  
        """)

    # Directory where images will be stored
    PREDICT_PATH = "toPredict"
    os.makedirs(PREDICT_PATH, exist_ok=True)

    IMAGE_PATH = os.path.join(PREDICT_PATH, "predictTree.png")

    # Layout for better organization
    col1, col2 = st.columns([1, 1])  

    with col1:  # Left side: Drawing Canvas
        st.subheader("🖌️ Draw Your Tree Here")
        st.write("Use the canvas below to draw a Tree. Once finished, save it before predicting.")

        canvas_result = st_canvas(
            fill_color="rgba(255,255,255,1)",  # White background
            stroke_width=3,
            stroke_color="black",
            background_color="white",
            height=224,
            width=224,
            drawing_mode="freedraw",
            key="canvas"
        )

        if st.button("💾 Save Drawing"):
            if canvas_result.image_data is not None:
                img_array = np.array(canvas_result.image_data)  # Convert canvas data to NumPy array

                if np.all(img_array[:, :, :3] == 255):  # Check if all pixels are white
                    st.warning("⚠️ Your canvas is empty! Please draw before saving.")
                else:
                    save_canvas_image(img_array)
                    st.success("✅ Drawing saved successfully!")

    with col2:  # Right side: Upload Image Section
        st.subheader("📤 Upload a Tree Drawing")
        st.write("If you've already drawn a Tree on paper, you can upload a scanned image here.")

        uploaded_file = st.file_uploader("Choose an image (PNG, JPG, JPEG)...", type=["png", "jpg", "jpeg"])

    # Prediction function using ENSEMBLE LEARNING (MAJORITY VOTING)
    def predict_image(image_path):
        """
        Pass the saved image to the models and return the majority prediction.
        """
        treePredict = [
            predict("model/tree/tree_model_10.tar", image_path),
            predict("model/tree/tree_model_12.tar", image_path),
            predict("model/tree/tree_model_15.tar", image_path),
        ]

        final_prediction = Counter(treePredict).most_common(1)[0][0]
        return final_prediction

    # Prediction Section
    st.subheader("🔍 Get Your Result")
    st.write("Once you have either drawn a Tree or uploaded an image, click the 'Predict' button below.")

    if st.button("🔍 Predict"):
        image_path = None

        if uploaded_file is not None:
            # Save uploaded image as the new input
            image = Image.open(uploaded_file)
            image.save(IMAGE_PATH)
            image_path = IMAGE_PATH
        elif os.path.exists(IMAGE_PATH):
            # If no upload, use the saved canvas image (if available)
            image_path = IMAGE_PATH

        # Process and predict
        if image_path:
            col1, col2 = st.columns([1, 2])  # Adjusting proportions for better readability

            with col1:  # Left side: Processed Image
                st.image(image_path, caption="🖼 Processed Image", width=210)  # Reduced size for a cleaner look

            with col2:  # Right side: Prediction Result & Explanation
                result = predict_image(image_path)

                feature_based_reports = {
                        0: {
                            "title": "❌ **Prediction Result: Signs of Depression & Low Energy**",
                            "traits": ["Depression", "Low Energy"],
                            "features": [
                                "Downward facing branches",
                                "Thin trunk",
                                "Absence of leaves"
                            ],
                            "base_report": (
                                "The drawing suggests signs of **low energy and possible depressive tendencies.** "
                                "Children experiencing such emotions may depict {features}, indicating a **sense of sadness, withdrawal, or lack of vitality.** "
                                "Downward-facing branches and the absence of leaves might symbolize **emotional exhaustion or difficulty in self-expression.** "
                                "Observing the child's **behavior, energy levels, and emotional responses** can provide further insights into their well-being."
                            ),
                        },
                        1: {
                            "title": "✅ **Prediction Result: Introverted & Low Ego Strength**",
                            "traits": ["Introverted", "Low Ego Strength"],
                            "features": [
                                "Short or no branches",
                                "Thin and small trunk"
                            ],
                            "base_report": (
                                "The drawing indicates a **reserved, introspective personality with potential low ego strength.** "
                                "The presence of {features} suggests a **lack of confidence, hesitation in social interactions, or emotional vulnerability.** "
                                "A short or absent branching structure might symbolize **limited outward expression and possible self-doubt.** "
                                "Providing **support, encouragement, and opportunities for self-expression** can help foster a sense of security and confidence in the child."
                            ),
                        },
                        2: {
                            "title": "🟡 **Prediction Result: Extroverted & Ambitious Personality**",
                            "traits": ["Extroverted", "Ambitious", "High Ego Strength"],
                            "features": [
                                "Thick trunk",
                                "Upward facing branches",
                                "Large number of branches"
                            ],
                            "base_report": (
                                "The drawing reflects a **confident, ambitious, and socially expressive personality.** "
                                "Children with extroverted tendencies may depict {features}, indicating **strong emotional resilience, enthusiasm, and a desire for growth.** "
                                "A thick trunk and numerous branches may symbolize **stability, ambition, and a willingness to engage with the environment.** "
                                "Such children may be **energetic, goal-oriented, and open to new experiences.** "
                                "Encouraging **creative activities and leadership opportunities** can help nurture their dynamic nature."
                            ),
                        }
                    }

                
                generate_report(result)

    if "final_report" in st.session_state and st.session_state["final_report"]:
        pdf_buffer = generate_pdf(st.session_state["final_report"], IMAGE_PATH)

        # Provide download button
        st.download_button(
            label="Download Report",
            data=pdf_buffer,
            file_name="psychological_report.pdf",
            mime="application/pdf"
        )
    else:
        st.error("No report generated yet. Please generate a report first.")


# Person Drawing Analysis Page
if selected == 'Person Drawing Test':
    st.title("🧑🏻‍🦱 Person Drawing Test")
    st.markdown("""
        ## Introduction  
        The **Person Test** is a psychological assessment that analyzes a child's emotions, self-perception, and personality through their drawing of a person. Our AI system evaluates features like proportions, details, and posture to classify emotional traits. An explainable AI (XAI) module provides insights into the reasoning behind the results, helping parents understand their child's psychological state.  

        ## Important Instructions for the Drawing Analysis  
        - **Stay Calm & Focused** – Ensure the child is in a normal, undisturbed mood before drawing, as this is a serious psychological assessment.  
        - **Create a 2D Drawing** – The person should be a simple two-dimensional sketch, not a 3D perspective.  
        - **Include Key Features** – The drawing can include facial features, hands, feet, clothing, or any other details.  
        - **Shading is Allowed** – Any part of the drawing can be shaded if desired.  
        - **Use Only Black Ink** – The drawing should be made with black ink or a black digital pen for consistency in analysis.  
        """)


    # Directory where images will be stored
    PREDICT_PATH = "toPredict"
    os.makedirs(PREDICT_PATH, exist_ok=True)

    IMAGE_PATH = os.path.join(PREDICT_PATH, "predictPerson.png")

    # Layout for better organization
    col1, col2 = st.columns([1, 1])  # Two equal columns

    with col1:  # Left side: Drawing Canvas
        st.subheader("🖌️ Draw Your Person Here")
        st.write("Use the canvas below to draw a person. Once finished, save it before predicting.")

        canvas_result = st_canvas(
            fill_color="rgba(255,255,255,1)",  
            stroke_width=3,
            stroke_color="black",
            background_color="white",
            height=224,
            width=224,
            drawing_mode="freedraw",
            key="canvas"
        )

        if st.button("💾 Save Drawing"):
            if canvas_result.image_data is not None:
                img_array = np.array(canvas_result.image_data)  # Convert canvas data to NumPy array

                if np.all(img_array[:, :, :3] == 255):  # Check if all pixels are white
                    st.warning("⚠️ Your canvas is empty! Please draw before saving.")
                else:
                    save_canvas_image(img_array)
                    st.success("✅ Drawing saved successfully!")

    with col2:  # Right side: Upload Image Section
        st.subheader("📤 Upload a person drawing")
        st.write("If you've already drawn a person on paper, you can upload a scanned image here.")

        uploaded_file = st.file_uploader("Choose an image (PNG, JPG, JPEG)...", type=["png", "jpg", "jpeg"])

    # Prediction function
    def predict_image(image_path):
        """
        Pass the saved image to the models and return the majority prediction.
        """
        personPredict = [
            predict("model/person/person_model_10.tar", image_path),
            predict("model/person/person_model_12.tar", image_path),
            predict("model/person/person_model_15.tar", image_path),
        ]

        final_prediction = Counter(personPredict).most_common(1)[0][0]
        return final_prediction

    # Prediction Section
    st.subheader("🔍 Get Your Result")
    st.write("Once you have either drawn a person or uploaded an image, click the 'Predict' button below.")

    if st.button("🔍 Predict"):
        image_path = None

        if uploaded_file is not None:
            # Save uploaded image as the new input
            image = Image.open(uploaded_file)
            image.save(IMAGE_PATH)
            image_path = IMAGE_PATH
        elif os.path.exists(IMAGE_PATH):
            # If no upload, use the saved canvas image (if available)
            image_path = IMAGE_PATH

        # Process and predict
        if image_path:
            col1, col2 = st.columns([1, 2])  # Adjusting proportions for better readability

            with col1:  # Left side: Processed Image
                st.image(image_path, caption="🖼 Processed Image", width=210)  # Reduced size for a cleaner look

            with col2:  # Right side: Prediction Result & Explanation
                result = predict_image(image_path)

                feature_based_reports = {
                        0: {
                            "title": "❌ **Psychological Indicator: Depression & Low Energy**",
                            "traits": ["Depression", "Low Energy", "Emotional Fatigue"],
                            "features": [
                                "Miniature size",
                                "Lack of detail",
                                "Faint or weak strokes"
                            ],
                            "base_report": (
                                "The child's drawing suggests **emotional exhaustion, low energy, or potential depressive tendencies.** "
                                "Common indicators include {features}, which may reflect **a sense of insignificance, withdrawal, or inner sadness.** "
                                "In psychological studies, small-sized figures often correlate with **low self-esteem, feelings of powerlessness, or insecurity.** "
                                "Additionally, weak or faded strokes may indicate **a lack of confidence or internal distress.** "
                                "It is recommended to observe the child's **social interactions, daily behavior, and emotional responses** to assess well-being."
                            ),
                        },
                        1: {
                            "title": "✅ **Psychological Indicator: Introversion & Social Withdrawal**",
                            "traits": ["Social Withdrawal", "Lack of Motivation", "Boredom", "Emotional Reservedness"],
                            "features": [
                                "Overly simplistic form",
                                "Significant lack of detail",
                                "Stick-figure representation"
                            ],
                            "base_report": (
                                "This drawing reflects a **reserved, introspective, and emotionally cautious personality.** "
                                "Children exhibiting {features} may show **social hesitation, difficulty expressing emotions, or preference for isolation.** "
                                "Psychological research suggests that **simplistic or absent facial features** may indicate **a reluctance to engage socially or difficulty in self-expression.** "
                                "Stick-figure drawings with minimal detail are often associated with **low motivation, feelings of loneliness, or emotional disengagement.** "
                                "Encouraging **creative storytelling, interactive play, and positive reinforcement** can help build the child’s confidence in self-expression."
                            ),
                        },
                        2: {
                            "title": "🟡 **Psychological Indicator: Anxiety & Obsessive Thinking**",
                            "traits": ["Anxiety", "Obsession", "Perfectionism", "Emotional Intensity"],
                            "features": [
                                "Highly detailed facial features",
                                "Proportional dimensions of limbs and body",
                                "Emphasis on structure and precision"
                            ],
                            "base_report": (
                                "The drawing reveals a **highly structured, detail-oriented, and possibly anxious mindset.** "
                                "Children who display {features} often exhibit **heightened attention to detail, perfectionist tendencies, or excessive self-monitoring.** "
                                "Psychologists note that highly detailed drawings may indicate **a desire for control, fear of imperfection, or increased cognitive processing related to anxiety.** "
                                "A strong focus on proportions and structure may suggest **intellectual maturity but could also reflect internal pressure to meet expectations.** "
                                "Encouraging **creative freedom, relaxation techniques, and self-expressive activities** can help ease potential anxiety and promote a balanced emotional outlook."
                            ),
                        }
                    }

                generate_report(result)


    if "final_report" in st.session_state and st.session_state["final_report"]:
        pdf_buffer = generate_pdf(st.session_state["final_report"], IMAGE_PATH)

        # Provide download button
        st.download_button(
            label="Download Report",
            data=pdf_buffer,
            file_name="psychological_report.pdf",
            mime="application/pdf"
        )
    else:
        st.error("No report generated yet. Please generate a report first.")


# Color Drawing Analysis Page
if selected == 'Color Drawing Test':
    st.title("Colored Image Drawing Test")
    st.markdown("""
        ## Introduction  
        The **Colored Image Test** is a psychological assessment that analyzes a child's emotions, self-perception, and personality through their drawing. Our AI system evaluates features like colors, proportions, details, and posture to classify emotional traits. An explainable AI (XAI) module provides insights into the reasoning behind the results, helping parents understand their child's psychological state.  

        ## Important Instructions for the Drawing Analysis  
        - **Stay Calm & Focused** – Ensure the child is in a normal, undisturbed mood before drawing, as this is a serious psychological assessment.  
        - **Create a 2D Drawing** – The image should be a simple two-dimensional sketch, not a 3D perspective.  
        - **Use Colors Freely** – The drawing should include colors, as color choices play a role in emotional analysis.  
        - **Include Key Features** – The drawing can include facial features, hands, feet, clothing, or any other details.  
        - **Shading is Allowed** – Any part of the drawing can be shaded if desired.  
        """)



    # Directory where images will be stored
    PREDICT_PATH = "toPredict"
    os.makedirs(PREDICT_PATH, exist_ok=True)

    IMAGE_PATH = os.path.join(PREDICT_PATH, "predictColor.png")

    # Layout for better organization
    col1, col2 = st.columns([1, 1])  # Two equal columns

    with col1:  # Left side: Drawing Canvas
        st.subheader("🖌️ Draw Here")
        st.write("Use the canvas below to draw. Once finished, save it before predicting.")

        canvas_result = st_canvas(
            fill_color="rgba(255,255,255,1)",  
            stroke_width=3,
            stroke_color="black",
            background_color="white",
            height=224,
            width=224,
            drawing_mode="freedraw",
            key="canvas"
        )

        if st.button("💾 Save Drawing"):
            if canvas_result.image_data is not None:
                img_array = np.array(canvas_result.image_data) 

                if np.all(img_array[:, :, :3] == 255):  
                    st.warning("⚠️ Your canvas is empty! Please draw before saving.")
                else:
                    save_canvas_image(img_array)
                    st.success("✅ Drawing saved successfully!")

    with col2:  
        st.subheader("📤 Upload a colored drawing")
        st.write("If you've already drawn on paper, you can upload a scanned image here.")

        uploaded_file = st.file_uploader("Choose an image (PNG, JPG, JPEG)...", type=["png", "jpg", "jpeg"])

    # Prediction function
    def predict_image(image_path):
        """
        Pass the saved image to the models and return the majority prediction.
        """
        colorPredict = [
            color_predict("model/color/color_model_20.tar", image_path),
            color_predict("model/color/color_model_10.tar", image_path),
            color_predict("model/color/color_model_40.tar", image_path),
        ]

        final_prediction = Counter(colorPredict).most_common(1)[0][0]
        return final_prediction

    # Prediction Section
    st.subheader("🔍 Get Your Result")
    st.write("Once you have either drawn a image or uploaded an image, click the 'Predict' button below.")

    if st.button("🔍 Predict"):
        image_path = None

        if uploaded_file is not None:
            # Save uploaded image as the new input
            image = Image.open(uploaded_file)
            image.save(IMAGE_PATH)
            image_path = IMAGE_PATH
        elif os.path.exists(IMAGE_PATH):
            # If no upload, use the saved canvas image (if available)
            image_path = IMAGE_PATH

        # Process and predict
        if image_path:
            col1, col2 = st.columns([1, 2])  # Adjusting proportions for better readability

            with col1:  # Left side: Processed Image
                st.image(image_path, caption="🖼 Processed Image", width=210)  # Reduced size for a cleaner look

            with col2:  # Right side: Prediction Result & Explanation
                result = predict_image(image_path)

                feature_based_reports = {
                    0: {
                        "title": "❌ **Psychological Insight: Expressions of Sadness & Emotional Reflection**",
                        "traits": ["Sadness", "Introspection", "Thoughtfulness", "Emotional Sensitivity"],
                        "features": [
                            "Soft or faint strokes",
                            "Minimal details in the drawing",
                            "Figures with downward-facing posture"
                        ],
                        "base_report": (
                            "This drawing may reflect **introspective emotions, temporary sadness, or a thoughtful mood.** "
                            "Common indicators such as {features} could suggest a child who is deeply reflective or experiencing a quieter emotional state. "
                            "Psychological research suggests that subtle or downward-facing strokes might correspond with **moments of emotional processing, a need for comfort, or quiet contemplation.** "
                            "Encouraging **open conversations, creative expression, and engaging activities** can provide emotional support and help the child express feelings in a positive way."
                        ),
                    },
                    1: {
                        "title": "✅ **Psychological Insight: Happiness & Emotional Positivity**",
                        "traits": ["Happiness", "Optimism", "Confidence", "Expressiveness"],
                        "features": [
                            "Bold and energetic strokes",
                            "Balanced and well-defined figures",
                            "Smiling expressions or lively imagery"
                        ],
                        "base_report": (
                            "This drawing suggests **joy, confidence, and an open emotional expression.** "
                            "Children who use {features} often exhibit **enthusiasm, curiosity, and a positive outlook on their environment.** "
                            "Research indicates that **vibrant strokes and expressive figures** often reflect **emotional stability and a strong sense of self-awareness.** "
                            "Encouraging **creative play, social interactions, and storytelling activities** can further nurture their confidence and emotional well-being."
                        ),
                    },
                    2: {
                        "title": "🟡 **Psychological Insight: Fear & Emotional Sensitivity**",
                        "traits": ["Cautiousness", "Sensitivity", "Uncertainty", "Emotional Awareness"],
                        "features": [
                            "Shaky or uneven strokes",
                            "Figures appearing small or hesitant",
                            "Use of darker shades or multiple erasures"
                        ],
                        "base_report": (
                            "This drawing may indicate **a cautious nature, emotional sensitivity, or a need for reassurance.** "
                            "Children expressing {features} may be navigating **new experiences, self-doubt, or a phase of emotional adjustment.** "
                            "Psychologists suggest that **shaky strokes and tentative figures** could reflect **internal hesitation or a desire for comfort and security.** "
                            "Providing **gentle encouragement, structured activities, and a supportive space** can help build confidence and ease any uncertainties they may feel."
                        ),
                    },
                    3: {
                        "title": "🔴 **Psychological Insight: Strong Emotions & Expressive Energy**",
                        "traits": ["Passion", "Intensity", "Determination", "Emotional Expression"],
                        "features": [
                            "Strong, forceful strokes",
                            "Exaggerated or bold features",
                            "Sharp angles or heavy lines"
                        ],
                        "base_report": (
                            "This drawing reflects **intense emotions, assertiveness, or strong personal expression.** "
                            "Children using {features} may be displaying **passion, determination, or an outlet for emotional energy.** "
                            "Studies suggest that **bold strokes and exaggerated features** often correspond with **strong-willed personalities or moments of emotional release.** "
                            "Encouraging **mindful activities, problem-solving discussions, and expressive storytelling** can help channel emotions into constructive and positive self-expression."
                        ),
                    }
                }


                generate_report(result)


    if "final_report" in st.session_state and st.session_state["final_report"]:
        pdf_buffer = generate_pdf(st.session_state["final_report"], IMAGE_PATH)

        # Provide download button
        st.download_button(
            label="Download Report",
            data=pdf_buffer,
            file_name="psychological_report.pdf",
            mime="application/pdf"
        )
    else:
        st.error("No report generated yet. Please generate a report first.")