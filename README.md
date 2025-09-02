# 🤟 Sign Language to Text Conversion using CNN + BBO

## 📌 Overview
This project implements a **real-time Sign Language to Text Converter** using **Convolutional Neural Networks (CNN)** optimized with **Biogeography-Based Optimization (BBO)**.  
It enables communication by converting hand gestures (ASL signs) into text with high accuracy.  

The system also supports **real-time translation** using **OpenCV**, making it useful for accessibility and human-computer interaction.  

---

## 🛠 Tech Stack
- Python  
- TensorFlow / Keras  
- OpenCV  
- NumPy, Pandas  
- Matplotlib  

---

## 📂 Dataset
- American Sign Language (ASL) image dataset.  
- Each image corresponds to a letter or symbol.  
- Preprocessing steps included:  
  - Grayscale conversion  
  - Resizing to fixed dimensions (e.g., 64×64)  
  - Normalization (pixel values scaled 0–1)  
  - Data augmentation (rotation, flipping, zooming)  

👉 If dataset cannot be uploaded, mention a source like [Kaggle ASL dataset](https://www.kaggle.com/datasets/datamunge/sign-language-mnist).  

---

## ⚙️ Workflow
1. **Data Preprocessing**  
   - Convert raw images → normalized grayscale images  
   - Apply augmentation to improve generalization  

2. **Model Building**  
   - CNN architecture for gesture recognition  
   - Optimization using **Biogeography-Based Optimization (BBO)**  

3. **Real-time Detection**  
   - Capture camera feed with OpenCV  
   - Predict gestures → convert to text  

---

## 🏗️ CNN Architecture
The **Convolutional Neural Network (CNN)** used in this project follows a simple yet powerful architecture:

- **Input Layer** → 64×64 grayscale image  
- **Conv2D + ReLU** → extract low-level features  
- **MaxPooling2D** → reduce spatial dimensions  
- **Conv2D + ReLU** → deeper feature extraction  
- **MaxPooling2D** → down-sampling  
- **Flatten** → convert feature maps to vector  
- **Dense (Fully Connected)** → classification layer with softmax  

Input Image → Conv2D → ReLU → Pooling → Conv2D → ReLU → Pooling → Flatten → Dense → Output (Letter)

![CNN Architecture](images/cnn_architecture.png)

---

## 🌍 Biogeography-Based Optimization (BBO)
**BBO** is an evolutionary optimization algorithm inspired by the **migration of species between habitats**.  
- Each **solution** = a “habitat” with a **Habitat Suitability Index (HSI)** (quality of the solution).  
- Good habitats (better solutions) share features with poor habitats through **migration**.  
- Random changes (mutations) maintain diversity and avoid local minima.  

🔹 In this project, BBO is used to:  
- Optimize **CNN hyperparameters** (learning rate, batch size, number of filters, etc.)  
- Improve **convergence speed** and **accuracy**  
- Prevent overfitting by balancing exploration (diversity) and exploitation (refinement)  

This makes the CNN model more efficient and robust for gesture recognition.

---

## 🌍 Biogeography-Based Optimization (BBO) Architecture

The BBO algorithm models the process of **species migration between habitats**:

1. **Initialization**  
   - Create a population of solutions (habitats).  
   - Each habitat = candidate CNN hyperparameters.  

2. **Evaluation**  
   - Calculate Habitat Suitability Index (HSI) → model accuracy or loss.  

3. **Migration**  
   - Good habitats (high HSI) share features with poor habitats.  
   - Parameters migrate to improve weak solutions.  

4. **Mutation**  
   - Randomly change some parameters to maintain diversity.  

5. **Termination**  
   - Repeat migration and mutation until best solution found.  

![BBO Flow](images/bbo_flow.png)

---

## 📊 Results
- Achieved **high accuracy (~99%)** on the ASL dataset using CNN + BBO  
- Successfully converted ASL gestures into text in **real time**  
- System runs efficiently on CPU and GPU  

---

## 🧑‍💻 How to Run

1. Clone the repository:
  ```bash
  git clone https://github.com/riyakansal04/sign-language-to-text.git
  cd sign-language-to-text

2. Install dependencies:
    ```bash
    pip install -r requirements.txt

3. Run the Jupyter Notebook:
    ```bash
    jupyter notebook sign_language.ipynb

---

📌 Future Work
1. Extend to word-level recognition (not just letters)
2. Deploy as a mobile app for accessibility
3. Integrate with speech synthesis for spoken output

---

## 👩‍💻 Author
**Riya Kansal**  
[LinkedIn](https://www.linkedin.com/in/riya-kansal-963042268/) • [GitHub](https://github.com/riyakansal04)
