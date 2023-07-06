# ========================================
# FileName: opencv_tools.py
# Date: 29 june 2023 - 08:00
# Author: Alexandre Benoit
# Email: alexandre.benoit@univ-smb.fr
# GitHub: https://github.com/albenoit/DeepLearningTools
# Brief: A module to custom plot
# for DeepLearningTools.
# =========================================

import cv2

def add_text_overlay(img, text, pos, scale=0.5, font_color=(255, 255, 255), bg_color= (0, 0, 0)):
   """
   Add a text overlay to an image.

   This function adds a text overlay to the given image. It uses the OpenCV library to draw a rectangle as the background
   for the text and then adds the text on top of it.

   The text overlay is placed at the specified position (top-left corner) on the image. The scale parameter can be used
   to adjust the size of the text. The font_color parameter determines the color of the text, and the bg_color parameter
   determines the background color of the text box.

   Note: The image is modified in-place. No new image is created.

   :param img: Image to which the text overlay will be added.
   :param text: Text to be displayed.
   :param pos: Position of the top-left corner of the text overlay.
   :param scale: Scale factor for the text size (default: 0.5).
   :param font_color: Color of the text (default: white).
   :param bg_color: Background color of the text box (default: black).

   :return: None
   """
   font_face = cv2.FONT_HERSHEY_SIMPLEX
   thickness = cv2.FILLED
   margin = 2
   txt_size = cv2.getTextSize(text, font_face, scale, thickness)

   end_x = pos[0] + txt_size[0][0] + margin
   end_y = pos[1] - txt_size[0][1] - margin

   cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
   cv2.putText(img, text, pos, font_face, scale, font_color, 1, cv2.LINE_AA)