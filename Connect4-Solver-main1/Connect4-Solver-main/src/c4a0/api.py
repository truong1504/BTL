from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
from typing import List
import random
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from c4a0.pos import Pos
from c4a0.training import TrainingGen
from c4a0.main import MCTS

app = FastAPI()

class GameState(BaseModel):
    board: List[List[int]]
    current_player: int
    valid_moves: List[int]

class AIResponse(BaseModel):
    move: int
    policy: List[float] = None

recent_ai_moves = []

def is_winning_move(board, column, player):
    rows = len(board)
    cols = len(board[0])
    
    row = -1
    for r in range(rows-1, -1, -1):
        if board[r][column] == 0:
            row = r
            break
    
    if row == -1:  
        return False
    
    test_board = [row[:] for row in board]
    test_board[row][column] = player
    
    def check_horizontal():
        for c in range(cols-3):
            if c <= column <= c+3:  
                if test_board[row][c] == player and test_board[row][c+1] == player and \
                   test_board[row][c+2] == player and test_board[row][c+3] == player:
                    return True
        return False
                
    def check_vertical():
        if row <= rows-4:  
            if test_board[row][column] == player and test_board[row+1][column] == player and \
               test_board[row+2][column] == player and test_board[row+3][column] == player:
                return True
        return False
    
    def check_diagonal_down():
        for r in range(rows-3):
            for c in range(cols-3):
                if r <= row <= r+3 and c <= column <= c+3:  
                    if test_board[r][c] == player and test_board[r+1][c+1] == player and \
                       test_board[r+2][c+2] == player and test_board[r+3][c+3] == player:
                        return True
        return False
    
    def check_diagonal_up():
        for r in range(3, rows):
            for c in range(cols-3):
                if r-3 <= row <= r and c <= column <= c+3:  
                    if test_board[r][c] == player and test_board[r-1][c+1] == player and \
                       test_board[r-2][c+2] == player and test_board[r-3][c+3] == player:
                        return True
        return False
    
    return check_horizontal() or check_vertical() or check_diagonal_down() or check_diagonal_up()

@app.post("/api/connect4-move")
async def make_move(game_state: GameState) -> AIResponse:
    global recent_ai_moves
    
    try:
        if not game_state.valid_moves:
            raise ValueError("No valid moves available")
        
        for move in game_state.valid_moves:
            if is_winning_move(game_state.board, move, game_state.current_player):
                recent_ai_moves.append(move)
                return AIResponse(move=move)
        
        opponent = 3 - game_state.current_player  
        for move in game_state.valid_moves:
            if is_winning_move(game_state.board, move, opponent):
                recent_ai_moves.append(move)
                return AIResponse(move=move)
        
        pos = Pos.from_2d_board(game_state.board)
        
        training_dir = "training"
        gen = TrainingGen.load_latest(training_dir)
        model = gen.get_model(training_dir)
        
        mcts = MCTS(pos, model, c_exploration=5.0, c_ply_penalty=0.01)
        mcts.run(2000) 
        
        policy = mcts.get_root_policy()
        
        valid_move_scores = {move: policy[move] for move in game_state.valid_moves}
        
        best_move = max(valid_move_scores, key=valid_move_scores.get)
        
        if len(recent_ai_moves) >= 3 and all(move == recent_ai_moves[-1] for move in recent_ai_moves[-3:]) and best_move == recent_ai_moves[-1]:

            alternative_moves = {m: s for m, s in valid_move_scores.items() if m != best_move}
            if alternative_moves:
                best_move = max(alternative_moves, key=alternative_moves.get)
        
        recent_ai_moves.append(best_move)
        if len(recent_ai_moves) > 10:
            recent_ai_moves.pop(0)
        
        return AIResponse(move=best_move, policy=[float(p) for p in policy])
    
    except Exception as e:
        if game_state.valid_moves:
            fallback_move = random.choice(game_state.valid_moves)
            recent_ai_moves.append(fallback_move)
            return AIResponse(move=fallback_move)
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)