import pygame
import numpy as np
from neural_network import NeuralNetwork
pygame.init()
win = pygame.display.set_mode((700,500))
pygame.display.set_caption('Number Recognition')
DRAW_WIDTH = DRAW_HEIGHT= 393
BUTTON_WIDTH = 139
BUTTON_HEIGHT = 49
BLACK = (0,0,0)
ORANGE = (232,152,48)
run = True
state = 1
output = []
pos_list = []
y_rect = [0 for _ in range(10)]
bar_count = 0
prev_pos = None
number = [pygame.image.load(f"image/number/{num}.png") for num in range(10)]
def test_button(pos):
   global state
   if(pos[0]in range(276,276+BUTTON_WIDTH) and pos[1]in range(423,423+BUTTON_HEIGHT) and state==1):
      if pygame.mouse.get_pressed()[0]:state=2
      win.blit(pygame.image.load("./image/test_button.png"),(276,423))
      rect = pygame.Surface((BUTTON_WIDTH, BUTTON_HEIGHT))
      rect.set_alpha(64)
      rect.fill((0,0,0))
      win.blit(rect,(276,423))
   else:win.blit(pygame.image.load("./image/test_button.png"),(276,423))
def reset_button(pos):
   global state, output, pos_list, y_rect, bar_count, prev_pos
   if(pos[0]in range(126,126+BUTTON_WIDTH) and pos[1]in range(423,423+BUTTON_HEIGHT)):
      if pygame.mouse.get_pressed()[0]:
         state = 1
         output = []
         pos_list = []
         y_rect = [0 for _ in range(10)]
         bar_count = 0
         prev_pos = None
      win.blit(pygame.image.load("./image/reset_button.png"),(126,423))
      rect = pygame.Surface((BUTTON_WIDTH, BUTTON_HEIGHT))
      rect.set_alpha(64)
      rect.fill((0,0,0))
      win.blit(rect,(126,423))
   else:win.blit(pygame.image.load("./image/reset_button.png"),(126,423))
def process_input():
   grid = [[0 for _ in range(28)] for _ in range(28)]
   xscale,yscale = DRAW_WIDTH//28, DRAW_HEIGHT//28
   for pos in pos_list:
      x,y = pos[0]//xscale, pos[1]//yscale
      if(x in range(28) and y in range(28)):
         grid[y][x] = 1
         if x + 1 < 28:grid[y][x+1] = 1
         if x - 1 >= 0:grid[y][x-1] = 1
         if y + 1 < 28:grid[y+1][x] = 1
         if y - 1 >= 0:grid[y-1][x] = 1
   return [pixel for line in grid for pixel in line]
while run:
   pos = pygame.mouse.get_pos()
   for event in pygame.event.get():
      if event.type == pygame.QUIT:run = False
   win.blit(pygame.image.load("./image/display.png"),(0,0))
   test_button(pos)
   for p in pos_list:pygame.draw.circle(win,BLACK,p,5)
   # drawing number state
   if(state==1):
      if(pos[0]in range(18,18+DRAW_WIDTH) and pos[1]in range(18,18+DRAW_HEIGHT)):
         pygame.mouse.set_visible(False)
         pygame.draw.circle(win,BLACK,pos,5)
         if pygame.mouse.get_pressed()[0]:
            print(len(pos_list))
            if(prev_pos is not None):
               step = max(abs(pos[0]-prev_pos[0]),abs(pos[1]-prev_pos[1]))
               for i in range(step):
                  x = int(prev_pos[0]+(pos[0]-prev_pos[0])*i/step)
                  y = int(prev_pos[1]+(pos[1]-prev_pos[1])*i/step)
                  if((x,y) not in pos_list):pos_list.append((x,y))
            if(pos not in pos_list):pos_list.append(pos)
            prev_pos = pos
         else:prev_pos = None
      else:pygame.mouse.set_visible(True)
   # processing drawn number and bar state
   elif(state==2):
      input_data = process_input()
      s = NeuralNetwork(784)
      s.add_hidden(64,'sigmoid')
      s.add_hidden(32,'sigmoid')
      s.add_output(10,'relu')
      s.load_weights('number.wb')
      # if(bar_count==1):
      #    accuracy = s.train(np.array([input_data]), np.array([np.array([0,0,0,0,0,0,0,1,0,0])]), 1, 0.01, return_value=['accuracy'], print_output=True)['accuracy']
      #    s.save_weights("number.wb")
      output = s.predict(input_data)
      max_num = np.max(output)
      min_num = np.min(output)
      output = np.int16(136*(output-min_num)/(max_num-min_num))
      if(bar_count==40):state = 3
      else:
         for i,j in enumerate(range(439,656,24)):
            y_rect[i]+=(output[i]/40)
            pygame.draw.rect(win,ORANGE,(j,187-y_rect[i],19,y_rect[i]))
         pygame.time.delay(20)
         bar_count+=1
   # result
   elif(state==3):
      for i,j in enumerate(range(439,656,24)):pygame.draw.rect(win,ORANGE,(j,187-y_rect[i],19,y_rect[i]))
      max_num_idx = np.argmax(output)
      win.blit(number[max_num_idx],(480,204))
      reset_button(pos)
   pygame.display.update()
pygame.quit()