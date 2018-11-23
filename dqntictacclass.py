import random
import numpy as np
import tensorflow as tf

player1 = 'x'
player2 = 'o'

def whoGoesFirst():

    if random.randint(0, 1) == 0:

        return 1

    else:

        return 2    

class agent():
     def __init__(self):
          
##     tf.reset_default_graph()
          self.inputs1 = tf.placeholder(shape=[None,27],dtype=tf.float32)
          self.W = tf.Variable(tf.random_uniform([27,9],0,0.01))
          self.Qout = tf.matmul(self.inputs1,self.W)
          self.predict = tf.argmax(self.Qout,1)

          self.nextQ = tf.placeholder(shape=[None,9],dtype=tf.float32)
          self.loss = tf.reduce_mean(tf.square(self.nextQ - self.Qout))
          self.updateModel = tf.train.GradientDescentOptimizer(0.1).minimize(self.loss)
          ##saver = tf.train.Saver()
          
class player():
     def __init__(self,turn):
          if turn == 1:
               self.char = 'x'
          else:
               self.char = 'o'
          available_actions = [0,1,2,3,4,5,6,7,8]
     def start_game(self):
          self.last_board = [' ']*9
          self.last_move = None
          move(self.last_board)

     def move(self,board):
          self.last_board = board
          
        
     def makeMove(self,board,action):
          self.last_move = action
          self.move_board = board
          self.move_board[self.last_move] = self.char
          return self.move_board 
     def converter(self,board):
           list1 = []
           for x in board:
               if x=='x':
                    list1.append(1)
               elif x =='o':
                    list1.append(-1)
               else:
                    list1.append(0)
           list1 = np.array(list1).reshape(1,9)
           self.new_board = np.zeros(27)
           for i in range(len(list1[0])):
               if(list1[0][i] == 0):
                    self.new_board[3 * i] = 1
               elif(list1[0][i] == 1):
                    self.new_board[3 * i + 1] = 1
               else:
                    self.new_board[3 * i + 2] = 1

           self.new_board = np.array(self.new_board).reshape(1,27)
           return(self.new_board)
     def play_a_game(self,state,action):
          state1 = np.copy(state)
          self.newBoard = self.makeMove(state1,action)
##          self.drawBoard(self.newBoard)
##          print(action)
##          print(state)
          self.reward = self.rewardget(self.newBoard,self.char,action,state)
          return self.newBoard,self.reward,self.donewith(self.newBoard,self.char,action,state)    

     def drawBoard(self,board):
          print('   |   |')

          print(' ' + board[6] + ' | ' + board[7] + ' | ' + board[8])

          print('   |   |')

          print('-----------')

          print('   |   |')

          print(' ' + board[3] + ' | ' + board[4] + ' | ' + board[5])

          print('   |   |')

          print('-----------')

          print('   |   |')

          print(' ' + board[0] + ' | ' + board[1] + ' | ' + board[2])

          print('   |   |')
          print("\n")

     def isWinner(self,bo, le):

         return ((bo[6] == le and bo[7] == le and bo[8] == le) or # across the top

         (bo[3] == le and bo[4] == le and bo[5] == le) or # across the middle

         (bo[0] == le and bo[1] == le and bo[2] == le) or # across the bottom

         (bo[6] == le and bo[3] == le and bo[0] == le) or # down the left side

         (bo[7] == le and bo[4] == le and bo[1] == le) or # down the middle

         (bo[8] == le and bo[5] == le and bo[2] == le) or # down the right side

         (bo[6] == le and bo[4] == le and bo[2] == le) or # diagonal

         (bo[8] == le and bo[4] == le and bo[0] == le)) # diagonal     

     def rewardget(self,board,player,action,oldboard):
     
          if self.isSpaceFree(oldboard,action) == False:
               return -90
          elif self.isSpaceFree(oldboard,action ) == True:
               if self.isWinner(board,player):
                    return 1
               elif self.isBoardFull(board):
                    return 0.5
               else:
                    return 0

     def donewith(self,board,player,action,oldboard):
         if self.isSpaceFree(oldboard,action) == False:
               return True
         elif self.isSpaceFree(oldboard,action ) == True:
               if self.isWinner(board,player):
                    return True
               elif self.isBoardFull(board):
                    return True
               else:
                    return False
          
     def isBoardFull(self,board):

         for i in range(0, 9):

             if self.isSpaceFree(board,i):

                 return False

         return True

     def isSpaceFree(self,board, move):

         return board[move] == ' '          
y = .99
e = 0.1
n0 = 100.0
start_size = 2
num_episodes = 100000
batch_size = 32
tf.reset_default_graph() #Clear the Tensorflow graph.
agent1 = agent()
agent2 = agent()
experience_1 = np.empty(0).reshape(0,27)
experience_2 = np.empty(0).reshape(0,27)
replay_1 = np.empty(0).reshape(0,9)
replay_2 = np.empty(0).reshape(0,9)

            

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    
    for i in range(num_episodes):
        if i >= start_size:
            e = max(n0 / (n0 + (i- start_size)), 0.1)
        else: e = 1
        available_actions = [0,1,2,3,4,5,6,7,8]
        done = False
        j = 0
        turn = whoGoesFirst()
        p1 = player(1)
        p2 = player(2)
        board = [' ']*9
        for max_steps in range(30):
            
            if turn == 1:

                p1.move(board)   
                board_converted_1 = p1.converter(board)
                action,allQ= sess.run([agent1.predict,agent1.Qout],feed_dict = {agent1.inputs1:board_converted_1})

                if np.random.rand(1)< e:
                    action[0] = random.choice(available_actions)
                
               
                newboard,reward,done = p1.play_a_game(p1.last_board,action[0])

                newboard_converted_1 = p1.converter(newboard)
                Q1 = sess.run(agent1.Qout,feed_dict={agent1.inputs1:newboard_converted_1})
                maxQ1 = np.max(Q1)
                targetQ = allQ
                targetQ[0,action[0]] = reward + y*maxQ1
                experience_1 = np.vstack([experience_1,board_converted_1])
                replay_1 = np.vstack([replay_1,targetQ])

                if reward == 1:
                     reward = -1
                     board_converted_2 = p2.converter(p2.last_board)
                     allQ= sess.run(agent2.Qout,feed_dict = {agent2.inputs1:board_converted_2})
                     newboard_converted_2 = p2.converter(p2.newBoard)

                     Q1 = sess.run(agent2.Qout,feed_dict={agent2.inputs1:newboard_converted_2})
                     maxQ1 = np.max(Q1)
                     targetQ = allQ
                     targetQ[0,p2.last_move] = reward + y*maxQ1
                     experience_2 = np.vstack([experience_2,board_converted_2])
                     replay_2 = np.vstack([replay_2,targetQ])
                     
                if reward == .5:
                     reward = .5
                     board_converted_2 = p2.converter(p2.last_board)
                     allQ= sess.run(agent2.Qout,feed_dict = {agent2.inputs1:board_converted_2})
                     newboard_converted_2 = p2.converter(p2.newBoard)

                     Q1 = sess.run(agent2.Qout,feed_dict={agent2.inputs1:newboard_converted_2})
                     maxQ1 = np.max(Q1)
                     targetQ = allQ
                     targetQ[0,p2.last_move] = reward + y*maxQ1
                     experience_2 = np.vstack([experience_2,board_converted_2])
                     replay_2 = np.vstack([replay_2,targetQ])
                
                board = newboard
                if done == True:
                    
                    break
                turn = 2
                

            else:

                p2.move(board)
                board_converted_2 = p2.converter(board)

                action,allQ= sess.run([agent2.predict,agent2.Qout],feed_dict = {agent2.inputs1:board_converted_2})

                if np.random.rand(1)< e:
                    action[0] = random.choice(available_actions)
                

                newboard,reward,done = p2.play_a_game(p2.last_board,action[0])


                newboard_converted_2 = p2.converter(newboard)
                Q1 = sess.run(agent2.Qout,feed_dict={agent2.inputs1:newboard_converted_2})
                maxQ1 = np.max(Q1)
                targetQ = allQ
                targetQ[0,action[0]] = reward + y*maxQ1
                experience_2 = np.vstack([experience_2,board_converted_2])
                replay_2 = np.vstack([replay_2,targetQ])
                
                if reward == 1:
                     reward = -1
                     board_converted_1 = p1.converter(p1.last_board)
                     allQ= sess.run(agent1.Qout,feed_dict = {agent1.inputs1:board_converted_1})
                     newboard_converted_1 = p1.converter(p1.newBoard)

                     Q1 = sess.run(agent1.Qout,feed_dict={agent1.inputs1:newboard_converted_1})
                     maxQ1 = np.max(Q1)
                     targetQ = allQ
                     targetQ[0,p1.last_move] = reward + y*maxQ1
                     experience_1 = np.vstack([experience_1,board_converted_1])
                     replay_1 = np.vstack([replay_1,targetQ])

                if reward == .5:
                     reward = .5
                     board_converted_1 = p1.converter(p1.last_board)
                     allQ= sess.run(agent1.Qout,feed_dict = {agent1.inputs1:board_converted_1})
                     newboard_converted_1 = p1.converter(p1.newBoard)

                     Q1 = sess.run(agent1.Qout,feed_dict={agent1.inputs1:newboard_converted_1})
                     maxQ1 = np.max(Q1)
                     targetQ = allQ
                     targetQ[0,p1.last_move] = reward + y*maxQ1
                     experience_1 = np.vstack([experience_1,board_converted_1])
                     replay_1 = np.vstack([replay_1,targetQ])

                
                board = newboard
                if done == True:
                    
                    break
                turn = 1

        if i % batch_size == 0:
            
            _,W1 =sess.run([agent1.updateModel,agent1.W],feed_dict={agent1.inputs1:experience_1,agent1.nextQ:replay_1})
            _,W1 =sess.run([agent2.updateModel,agent2.W],feed_dict={agent2.inputs1:experience_2,agent2.nextQ:replay_2})
            experience_1 = np.empty(0).reshape(0,27)
            experience_2 = np.empty(0).reshape(0,27)
            replay_1 = np.empty(0).reshape(0,9)
            replay_2 = np.empty(0).reshape(0,9)
##            print("not done")
            
        if i %5000 == 0:
            print("Episode no.:",i)
##        saver.save(sess,"./dqn_tic_tac.ckpt")
    final_weights_1 = sess.run(agent1.W)
    np.savez('weight_storage_1',final_weights_1)
    final_weights_2 = sess.run(agent2.W)
    np.savez('weight_storage_2',final_weights_2)

     
