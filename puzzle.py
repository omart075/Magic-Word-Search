import time
import argparse
import numpy as np
import cv2
import pytesseract
from PIL import Image

# global variables
x = []
y = []
rows = []
cols = []
wordPoints = []

def search2D(puzzle, row, col, word):
    ''' Helper function that searches puzzle in all 8 directions. '''

    global x, y, rows, cols, wordPoints
    #if first character of word does not match given starting point
    if puzzle[row][col].lower() != word[0].lower():
        return False

    #a safe copy of word as each letter is found
    checkWord = str(puzzle[row][col].lower())

    #print puzzle[row][col].lower()
    wordLen = len(word)

    for i in range(8):
        #reset temp if same letter is found more than once
        temp = checkWord

        #initialize starting point for current direction
        rd = row + x[i]
        cd = col + y[i]

        #first character is already checked; match remaining characters
        for k in range(1, wordLen):

            #if it goes out of bounds
            if rd >= rows or rd < 0 or cd >= cols or cd < 0:
                break

            #if not matched
            if str(puzzle[rd][cd].lower()) != word[k].lower():
                break

            #add each letter of the word as it is found
            temp += str(puzzle[rd][cd].lower())
            # keep track of all points for each letter in word
            wordPoints.append((rd, cd))

            #keep moving in same direction
            rd += x[i]
            cd += y[i]

        #only gets here if all letters in word were found
        if temp == word:
            return True

    return False


def searchWord(puzzle, word, rows, cols, wordCoordinates):
    '''
    Searches for individual words using an array made of rows of letters from
    the puzzle.
    '''

    for row in range(rows):
        for col in range(cols):
            if search2D(puzzle, row, col, word):
                wordCoordinates.append((row, col, wordPoints[len(wordPoints)-1]))
                return word + " found at: " + str((row, col)) + " -> " + str(wordPoints[len(wordPoints)-1])

    return "Word not found"

def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]

def get_bounding_rectangles(im):
    ''' Mozularize image by making rectangular blobs for each section of text. '''

    # find edges
    edges = cv2.Canny(im,100,200)

    # dilate to form blocks
    kernel = np.ones((4,4), np.uint8)
    dilation = cv2.dilate(edges, kernel, iterations=5)
    cv2.imshow('dilated', dilation)
    cv2.waitKey(0)

    # find contours
    ctrs, hier = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # get rectangles containing each contour
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]

    # sort from smallest to largest area    
    rects.sort(key=lambda x: ( (x[0] + x[2]) - x[0] )*( (x[1] + x[3]) - x[1] ))
    # list of areas
    rect_areas = [( (x[0] + x[2]) - x[0] )*( (x[1] + x[3]) - x[1] ) for x in rects]

    # find outliers
    data = np.asarray(rect_areas)
    data = reject_outliers(data).tolist()

    # filter outliers
    indeces = []
    for x in data:
        indeces.append(rect_areas.index(x))    
    filtered_rects = []
    for x in indeces:
        filtered_rects.append(rects[x])

    # append the puzzle, which should be the largest rectangle
    filtered_rects.append(rects[-1])
    rects = filtered_rects

    return rects


def convert_to_text(rects, im):
    '''
    Read text found in each rectangle of text and determine which rectangle is
    the puzzle and which are the words to find.
    '''

    compareVal = 0
    currentArea = 0
    words = []
    puzzle = []
    puzzleRect = ()
    for rect in rects:
        # calculate area of rectangles found to determine which is the puzzle/words
        length = (rect[0] + rect[2]) - rect[0]
        width = (rect[1] + rect[3]) - rect[1]
        currentArea = length * width

        # draw the rectangles
        cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
        cv2.imshow('blocks found', im)
        cv2.waitKey(0)

        # crop out each rectangle found
        crop = im[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
        height, width = crop.shape[:2]

        # resize for standarized data set
        if width < 200 or height < 200:
            crop = cv2.resize(crop, (400, 300))

        cv2.imwrite("imgs/crop.png", crop)
        crop = Image.fromarray(crop)

        # extract text from each block
        text = pytesseract.image_to_string(crop, config="-psm 6").split('\n')
        text = [row.encode("utf-8").strip().replace(' ', '').lower() for row in text]

        if (currentArea - compareVal) < 100000:
            for word in text:
                if len(word) > 0:
                    words.append(word)

        else:
            puzzle = [row for row in text if row != '']
            puzzleRect = (rect[0], rect[1], rect[2], rect[3])

            break

    return words, puzzle, puzzleRect


def solve_puzzle(words, puzzle):
    ''' Convert puzzle and words into arrays and solve the puzzle. '''

    global x, y, rows, cols
    rows = len(puzzle)
    cols = len(puzzle[0])

    #array that identifies directions
    x = [-1, -1, -1, 0, 0, 1, 1, 1]
    y = [-1, 0, 1, -1, 1, -1, 0, 1]
    # array that keeps track of the point for each letter in the word
    wordCoordinates = []
    for word in words:
        print searchWord(puzzle, word, rows, cols, wordCoordinates)
    return wordCoordinates


def draw_lines(puzzleRect, wordCoordinates, im, puzzle):
    '''
    Map position of words found in array to their position in the image and
    draw lines where they were found.
    '''

    numRows = len(puzzle)
    numCols = len(puzzle[0])
    incrementX = puzzleRect[2]/numCols # width/cols
    incrementY = puzzleRect[3]/numRows # height/rows

    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    for coordinate in wordCoordinates:
        # x = origin + number of cols + half a column
        # y = origin + number of rows + half a row
        p1 = (puzzleRect[0] + (coordinate[1]*incrementX) + (incrementX/2), puzzleRect[1] + (coordinate[0]*incrementY) + (incrementY/2))
        p2 = (puzzleRect[0] + (coordinate[2][1]*incrementX) + (incrementX/2), puzzleRect[1] + (coordinate[2][0]*incrementY) + (incrementY/2))
        cv2.arrowedLine(im, p1, p2, (0, 0, 255), 1)
    cv2.imshow('solved puzzle', im)
    cv2.waitKey(0)


def main():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--puzzleImage", required=True,
    	help="name of puzzle image")
    args = vars(ap.parse_args())

    # read image with words to find and resize for standarized data set
    im = cv2.imread("imgs/" + args['puzzleImage'],0)   
    im = cv2.resize(im, (600, 800))

    # ret, im = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # cv2.imshow('binary', im)
    # cv2.waitKey(0)

    final_im = im.copy()

    rects = get_bounding_rectangles(im)

    words, puzzle, puzzleRect = convert_to_text(rects, im)

    # print words
    # print '\n'
    # for row in puzzle:
    #     print row
    # print '\n'

    wordCoordinates = solve_puzzle(words, puzzle)
    print("\nNumber of Words: {}".format(len(words)))
    print("Number of Words Found: {}".format(len(wordCoordinates)))

    draw_lines(puzzleRect, wordCoordinates, final_im, puzzle)



if __name__ == "__main__":
    main()

#TODO:
#    - check out markerless ar "rendering 3d object" commit" for 3d lines
