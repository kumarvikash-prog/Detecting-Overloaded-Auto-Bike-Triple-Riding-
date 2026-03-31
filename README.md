# 🛵 Catching Overloaded Bikes: A Beginner-Friendly Computer Vision Project

## What does this project do?

In many parts of the world, riding a two-wheeler (like a motorcycle or scooter) with more than two people is a dangerous traffic violation—often called "triple riding" or "overloading."

This project acts as an automatic traffic observer! You give it a picture of a bike, and it will:
1. **Look** for the people sitting on the vehicle.
2. **Count** how many people there are.
3. **Decide** if there is a violation: returning "Normal Riding" (1 or 2 people) or raising an alarm like "Triple Riding Detected" or "Severely Overloaded" (for 4+ people).

---

## How does it work? (Explained for Beginners!)

You don't need a math degree or an Artificial Intelligence background to understand this. Here is the step-by-step "magic" happening behind the scenes, explained in simple English:

### Step 1: Making the Image Easy to Read (Preprocessing)
First, the computer strips away all the colors from the image (making it black and white) and adjusts the lighting contrast. This makes it much easier for the computer to focus on important shapes and shadows without getting confused by bright, colorful clothes or neon signs.

### Step 2: The "Face Hunt" (Detection)
Instead of trying to find whole human bodies—which is very difficult when people are squeezed together on a small scooter—we look for **faces**! 

We use an old-school but highly effective tool called a **Haar Cascade**. Think of it as a transparent stencil that the computer slides across the image. It looks for a specific pattern of light and dark pixels that usually make up two eyes, a nose, and a mouth. We use two different stencils:
- One for **front-facing** people.
- One for **side-facing** people (a profile).

### Step 3: Throwing Away Mistakes (Filtering)
Sometimes, a pattern on a plaid shirt, a cluster of leaves, or a person sitting in a car in the background might accidentally match our "face stencil." We use simple, clever rules to throw out these mistakes:
- **The Size Rule:** If a "face" is way too tiny (like a pebble) or way too huge (taking up the whole screen), it's probably a mistake. We ignore it.
- **The Location Rule (Region of Interest):** People sitting on a bike will almost always have their heads in the **top half** of the picture. So, we tell the computer to completely ignore any "faces" it finds near the wheels, legs, or the road!

### Step 4: From Faces to Bodies (Expansion)
Once we are confident we have found a face, we can make a very safe guess about where the rest of the person's upper body is! The code automatically draws a larger rectangle that starts at the face and expands downward to cover the person's chest and shoulders.

### Step 5: Counting and Deciding
Finally, we carefully count the number of body rectangles we drew.
- **1 or 2 rectangles:** We print "**Normal Riding (2 People)**" in bright green.
- **3 rectangles:** We print "**Triple Riding Detected**" in warning red.
- **4 or more rectangles:** We print "**Severely Overloaded**" in warning red.

---

## How to Use This Project?

### 1. What You Need (Installation)
You just need Python installed on your computer, along with two very common libraries: `opencv-python` (which handles the image processing) and `numpy`. 

Here is how you can set it up on your specific operating system:

#### 🪟 For Windows Users
1. **Install Python:** Download and install Python from [python.org](https://www.python.org/downloads/). *(Important: Make sure to check the box that says "Add Python to PATH" during installation!)*
2. **Open Command Prompt:** Press the Windows key, type `cmd`, and press Enter.
3. **Navigate to the project:** Use the `cd` (change directory) command to go to the folder where you saved this project. For example:
   ```cmd
   cd C:\Users\YourName\Downloads\Detecting-Overloaded-Auto-Bike
   ```
4. **Install the libraries:** Run this command:
   ```cmd
   pip install -r requirements.txt
   ```

#### 🍎 For Mac Users
1. **Open Terminal:** Press `Command + Space`, type `Terminal`, and hit Enter.
2. **Navigate to the project:** Use the `cd` command to go to the folder where you saved this project. For example:
   ```bash
   cd ~/Downloads/Detecting-Overloaded-Auto-Bike
   ```
3. **Install the libraries:** Mac usually uses `pip3`. Run this command:
   ```bash
   pip3 install -r requirements.txt
   ```

#### 🐧 For Linux Users
1. **Open Terminal:** Press `Ctrl + Alt + T`.
2. **Navigate to the project:** Use the `cd` command to go to the folder where you saved this project:
   ```bash
   cd ~/Downloads/Detecting-Overloaded-Auto-Bike
   ```
3. **Install the libraries:** Run this command:
   ```bash
   pip3 install -r requirements.txt
   ```

### 2. Running the Tool
Now that everything is installed, you are ready to play with the code!

1. **Add your pictures:** Drop any test pictures of bikes you want to check into the `input/` folder.
2. **Run the magic command:** In your Command Prompt or Terminal, type the following:
   - On **Windows**, run: `python main.py`
   - On **Mac or Linux**, run: `python3 main.py`

The program will process all the images in the input folder and save the final results inside the `output/` folder! If you open those result images, you will see bright green boxes drawn over the riders and the total count written clearly in the corner.

---

## Limitations (When might it get confused?)
No system is perfect, especially simple ones! Here are a few real-world things that might trick the detector:
- **Full-Face Helmets:** If a rider is wearing a dark, fully-closed helmet with the visor down, the "face stencil" won't be able to see their eyes and nose, so they might not be counted.
- **Looking Away:** If someone has their back completely turned to the camera, the computer can't see their face to count them.
- **Very Blurry Images:** If the picture is taken at high speed on a bumpy road, the shapes become too smudged for the computer to recognize the "face" pattern.

---

## A Note for Absolute Beginners
This project does *not* use heavy, complicated "Deep Learning" or massive "AI Neural Networks" that require expensive computers and graphics cards. 

Instead, it uses **"Classical Computer Vision"**—which basically means we are using smart, lightweight pixel math and logical rules. It’s light, fast, and a perfect first step into the amazing world of teaching computers how to see!