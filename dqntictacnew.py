import random
import numpy as np
import tensorflow as tf

player1 = 'x'
player2 = 'o'
def drawBoard(board):
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


def isWinner(bo, le):

    return ((bo[6] == le and bo[7] == le and bo[8] == le) or # across the top

    (bo[3] == le and bo[4] == le and bo[5] == le) or # across the middle

    (bo[0] == le and bo[1] == le and bo[2] == le) or # across the bottom

    (bo[6] == le and bo[3] == le and bo[0] == le) or # down the left side

    (bo[7] == le and bo[4] == le and bo[1] == le) or # down the middle

    (bo[8] == le and bo[5] == le and bo[2] == le) or # down the right side

    (bo[6] == le and bo[4] == le and bo[2] == le) or # diagonal

    (bo[8] == le and bo[4] == le and bo[0] == le)) # diagonal

def rewardget(board,player,action,oldboard):
     
     if isSpaceFree(oldboard,action) == False:
          return -90
     elif isSpaceFree(oldboard,action ) == True:
          if isWinner(board,player):
               return 1
          elif isBoardFull(board):
               return 0.5
          else:
               return 0
          
     
def convert_state_representation(state):
    new_board = np.zeros(27)
    for i in range(len(state[0])):
            if(state[0][i] == 0):
                new_board[3 * i] = 1
            elif(state[0][i] == 1):
                new_board[3 * i + 1] = 1
            else:
                new_board[3 * i + 2] = 1

    return(new_board)


def donewith(board,player,action,oldboard):
    if isSpaceFree(oldboard,action) == False:
          return True
    elif isSpaceFree(oldboard,action ) == True:
          if isWinner(board,player):
               return True
          elif isBoardFull(board):
               return True
          else:
               return False
          
def isBoardFull(board):

    for i in range(0, 9):

        if isSpaceFree(board,i):

            return False

    return True

def isSpaceFree(board, move):

    return board[move] == ' '

def whoGoesFirst():

    if random.randint(0, 1) == 0:

        return 1

    else:

        return 2

def allpossiblemoves(state):
     return [i for i, e in enumerate(state) if e == " "]

def makeMove(board, letter, move):
     
    copyboard = board.copy()
    copyboard[move] = letter
    return copyboard

def convert_to_one_hot(state):
     list1 = []
     for x in state:
          if x=='x':
               list1.append(1)
          elif x =='o':
               list1.append(-1)
          else:
               list1.append(0)
     return list1

def convert_to_list(state):
     list2 = []
     for x in state:
          if x== 1 :
               list2.append('x')
          elif x == -1:
               list2.append('o')
          else:
               list2.append(' ')
     return list2

    
def play_a_game(state,action,player):
    state1 = state.copy()
    newBoard = makeMove(state,player,action)
##    drawBoard(newBoard)
    reward = rewardget(newBoard,player,action,state1)
    return newBoard,reward,donewith(newBoard,player,action,state1)
    


tf.reset_default_graph()
inputs1 = tf.placeholder(shape=[1,27],dtype=tf.float32)
W = tf.Variable(tf.random_uniform([27,9],0,0.01))
Qout = tf.matmul(inputs1,W)
predict = tf.argmax(Qout,1)

nextQ = tf.placeholder(shape=[1,9],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.02)
updateModel = trainer.minimize(loss)
##saver = tf.train.Saver()
init = tf.global_variables_initializer()

y = .99
e = 0.2
num_episodes = 200000
jList = []
rList = []
rAll = 0
with tf.Session() as sess:
    sess.run(init)
    
    for i in range(num_episodes):
##        print("episode no.:",i)
        
        d =False
        j = 0
        turn = 2
        board = [' ']*9
        actions = [0,1,2,3,4,5,6,7,8]
        s = np.array(convert_to_one_hot(board)).reshape(1,9)
        for max_steps in range(30):
            
            if turn == 1:
                s = convert_state_representation(s).reshape(1,27)
##                print("s is :",s)
                a,allQ,weights= sess.run([predict,Qout,W],feed_dict = {inputs1:s})
                previous_state = np.copy(s)
##                print("copy of s:",previous_state)
                if np.random.rand(1)< e:
##                    print("random action")
                    a[0] = random.choice(actions)
                
##                print("weights:",weights)
##                print("Qout",allQ)
               
##                print("action is",a[0])
                newboard,r,d = play_a_game(board,a[0],player1)
##                print("Newboard:",newboard)
##                print("reward is:",r)
                s1 = np.array(convert_to_one_hot(newboard)).reshape(1,9)
##                print("next state:",s1)
                s1 = convert_state_representation(s1).reshape(1,27)
                Q1 = sess.run(Qout,feed_dict={inputs1:s1})
                maxQ1 = np.max(Q1)
                targetQ = allQ
                targetQ[0,a[0]] = r + y*maxQ1
##                print("after :",targetQ)
##                s = np.array(convert_to_one_hot(s)).reshape(1,9)
                _,W1 =sess.run([updateModel,W],feed_dict={inputs1:s,nextQ:targetQ})
                final_weights = sess.run(W)
                np.savez('weight_storage',final_weights)
##                print("final weights:",final_weights)
##                print(weights == final_weights)
                rAll += r
                rList.append(rAll)

                s  = s1
                board = newboard
                if d == True:
                    
                    break
                turn = 2
                

            else:
                 
                 choices = allpossiblemoves(board)
                 action = random.choice(choices)
##                 print("Player 2 action:",action)
                 newboard,r,d = play_a_game(board,action,player2)
                 s1 = np.array(convert_to_one_hot(newboard)).reshape(1,9)
                 if r == 1:
                      r = -1
                      state = convert_state_representation(s1).reshape(1,27)
                      Q1 = sess.run(Qout,feed_dict={inputs1:s})
                      maxQ1 = np.max(Q1)
                      targetQ = allQ
                      targetQ[0,a[0]] = r + y*maxQ1
                      _,W1 =sess.run([updateModel,W],feed_dict={inputs1:previous_state,nextQ:targetQ})
                      final_weights = sess.run(W)
                      np.savez('weight_storage',final_weights)

                 if r == 0.5:
                      r = 0.5
                      state = convert_state_representation(s1).reshape(1,27)
                      Q1 = sess.run(Qout,feed_dict={inputs1:s})
                      maxQ1 = np.max(Q1)
                      targetQ = allQ
                      targetQ[0,a[0]] = r + y*maxQ1
                      _,W1 =sess.run([updateModel,W],feed_dict={inputs1:previous_state,nextQ:targetQ})
                      final_weights = sess.run(W)
                      np.savez('weight_storage',final_weights)     

                 s = s1
                 board = newboard
                 if d ==True:
                      break

                 rList.append(rAll)
                 turn = 1
        if i %1000 == 0:
            print("Episode no.:",i)
##        saver.save(sess,"./dqn_tic_tac.ckpt")
print("Percent of successful episodes:"+str(sum(rList)/num_episodes)+"%")

