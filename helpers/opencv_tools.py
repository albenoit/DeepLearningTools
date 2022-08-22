import cv2
def add_text_overlay(img, text, pos, scale=0.5, font_color=(255, 255, 255), bg_color= (0, 0, 0)):

   font_face = cv2.FONT_HERSHEY_SIMPLEX
   thickness = cv2.FILLED
   margin = 2
   txt_size = cv2.getTextSize(text, font_face, scale, thickness)

   end_x = pos[0] + txt_size[0][0] + margin
   end_y = pos[1] - txt_size[0][1] - margin

   cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
   cv2.putText(img, text, pos, font_face, scale, font_color, 1, cv2.LINE_AA)