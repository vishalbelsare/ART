import traceback
from pydantic import BaseModel
import art
from art.local import LocalBackend
from dotenv import load_dotenv
from openai import AsyncOpenAI
import openai
import asyncio
import json
import os
import random
from typing import List, Tuple

load_dotenv()

GAME_SYSTEM_PROMPT = """You are playing tic-tac-toe. You are X and you go first. The opponent is O.

Use the make_move tool to make your move by specifying row and column coordinates (0-2 for both).
Always use the tool to make your move - do not respond with text only."""

BOARD_UPDATE_TEMPLATE = """Current board after moves:
{board_str}

Game status: {game_status}"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "make_move",
            "description": (
                "Make a move on the tic-tac-toe board. The agent needs to specify the row and column coordinates (0-2 for both). "
                "The function returns the board after the move."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "row": {
                        "type": "integer",
                        "description": "The row coordinate (0-2).",
                    },
                    "col": {
                        "type": "integer",
                        "description": "The column coordinate (0-2).",
                    },
                },
                "required": ["row", "col"],
            },
        },
    }
]


class TicTacToeMove(BaseModel):
    row: int
    col: int


class TicTacToeBoard:
    def __init__(self):
        self.board = [[" " for _ in range(3)] for _ in range(3)]

    def is_valid_move(self, row: int, col: int) -> bool:
        return 0 <= row < 3 and 0 <= col < 3 and self.board[row][col] == " "

    def make_move(self, row: int, col: int, player: str) -> bool:
        if self.is_valid_move(row, col):
            self.board[row][col] = player
            return True
        return False

    def get_available_moves(self) -> List[Tuple[int, int]]:
        return [(r, c) for r in range(3) for c in range(3) if self.board[r][c] == " "]

    def check_winner(self) -> str | None:
        # Check rows
        for row in self.board:
            if row[0] == row[1] == row[2] != " ":
                return row[0]

        # Check columns
        for col in range(3):
            if self.board[0][col] == self.board[1][col] == self.board[2][col] != " ":
                return self.board[0][col]

        # Check diagonals
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != " ":
            return self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != " ":
            return self.board[0][2]

        return None

    def is_full(self) -> bool:
        return all(self.board[r][c] != " " for r in range(3) for c in range(3))

    def is_game_over(self) -> bool:
        return self.check_winner() is not None or self.is_full()

    def display(self) -> str:
        lines = []
        for i, row in enumerate(self.board):
            lines.append(" | ".join(row))
            if i < 2:
                lines.append("---------")
        return "\n".join(lines)

    def copy(self):
        new_board = TicTacToeBoard()
        new_board.board = [row[:] for row in self.board]
        return new_board


class RandomOpponent:
    def __init__(self):
        self.player = "O"

    def get_move(self, board: TicTacToeBoard) -> Tuple[int, int]:
        available_moves = board.get_available_moves()
        if available_moves:
            return random.choice(available_moves)
        raise ValueError("No available moves")


async def train():
    backend = LocalBackend()

    # Load model config from JSON file if it exists, otherwise use default
    config_path = "model_config.json"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            model_config = json.load(f)
        model = art.TrainableModel.model_validate(model_config)
        print(f"Loaded model config from {config_path}: {model.name}")
    else:
        model = art.TrainableModel(
            name="001",
            project="tic-tac-toe",
            base_model="Qwen/Qwen2.5-7B-Instruct",
            _internal_config=art.dev.InternalModelConfig(
                engine_args=art.dev.EngineArgs(gpu_memory_utilization=0.7),
            ),
        )
        print("Using default model config")
    await model.register(backend)

    def make_llm_move(board: TicTacToeBoard, row: int, col: int) -> tuple[str, bool]:
        """Make the LLM's move and return just the board state after that move.

        Args:
            board: The current board state
            row: Row coordinate (0-2)
            col: Column coordinate (0-2)
            game_history: List to track moves

        Returns:
            tuple[str, bool]: (board_after_move, move_was_invalid)
        """
        if not board.is_valid_move(row, col):
            return (
                f"Invalid move: position ({row},{col}) is already occupied or out of bounds",
                True,
            )

        # Make LLM's move
        board.make_move(row, col, "X")

        # Check if game is over after LLM's move
        winner = board.check_winner()
        if winner == "X":
            return f"You played ({row},{col}). You win!\n\n{board.display()}", False
        elif winner is None and board.is_full():
            return f"You played ({row},{col}). It's a draw!\n\n{board.display()}", False

        # Game continues, show board after LLM's move
        return f"You played ({row},{col}).\n\n{board.display()}", False

    def make_opponent_move(board: TicTacToeBoard) -> str:
        """Make the opponent's move and return the user message describing it.

        Args:
            board: The current board state (after LLM's move)
            game_history: List to track moves

        Returns:
            str: Message describing opponent's move and game state
        """
        if board.is_game_over():
            return "Game is already over."

        try:
            opponent = RandomOpponent()
            opp_move = opponent.get_move(board)
            board.make_move(opp_move[0], opp_move[1], "O")

            # Check if game is over after opponent's move
            winner = board.check_winner()
            if winner == "O":
                return f"Opponent played ({opp_move[0]},{opp_move[1]}). You lose!\n\n{board.display()}"
            elif winner is None and board.is_full():
                return f"Opponent played ({opp_move[0]},{opp_move[1]}). It's a draw!\n\n{board.display()}"

            return f"Opponent played ({opp_move[0]},{opp_move[1]}).\n\n{board.display()}\n\nYour move, play as tool call."

        except ValueError:
            return "Game over - no more moves available."

    async def rollout(client: openai.AsyncOpenAI) -> art.Trajectory | BaseException:
        try:
            board = TicTacToeBoard()
            move_invalid = False

            trajectory = art.Trajectory(
                messages_and_choices=[
                    {"role": "system", "content": GAME_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Let's play tic-tac-toe! You are X.\n\nCurrent board:\n{board.display()}\n\nMake your first move!",
                    },
                ],
                tools=TOOLS,
                reward=0,
            )

            max_turns = 10  # Prevent infinite loops
            turn = 0

            while not board.is_game_over() and turn < max_turns:
                turn += 1

                chat_completion = await client.chat.completions.create(
                    messages=trajectory.messages(),
                    model=model.name,
                    max_tokens=200,
                    timeout=60,
                    tools=TOOLS,
                )

                choice = chat_completion.choices[0]
                trajectory.messages_and_choices.append(choice)

                # Handle tool calls
                if choice.message.tool_calls:
                    for tool_call in choice.message.tool_calls:
                        if tool_call.function.name == "make_move":
                            try:
                                args = json.loads(tool_call.function.arguments)
                                row, col = args["row"], args["col"]

                                # Make LLM's move and get board state
                                result, is_invalid = make_llm_move(board, row, col)
                                if is_invalid:
                                    move_invalid = True

                                # Add tool response with board after LLM's move
                                trajectory.messages_and_choices.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": tool_call.id,
                                        "content": result,
                                    }
                                )

                                # If game isn't over and move was valid, make opponent's move
                                if not is_invalid and not board.is_game_over():
                                    opponent_msg = make_opponent_move(board)
                                    trajectory.messages_and_choices.append(
                                        {
                                            "role": "user",
                                            "content": opponent_msg,
                                        }
                                    )

                            except Exception:
                                move_invalid = True
                        else:
                            move_invalid = True
                else:
                    move_invalid = True

                if move_invalid:
                    break

            # Calculate reward
            winner = board.check_winner()
            if move_invalid:
                reward = 0.0  # Penalty for invalid moves
            elif winner == "X":
                reward = 1.0  # LLM wins
            elif winner == "O":
                reward = 0.0  # LLM loses
            else:
                reward = 0.5  # Draw

            trajectory.reward = reward
            return trajectory

        except Exception as e:
            print(f"Error in rollout: {e} -- {traceback.format_exc()}")
            return e

    openai_client = AsyncOpenAI(
        api_key=model.inference_api_key, base_url=model.inference_base_url
    )

    for _ in range(await model.get_step(), 1_000):
        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(rollout(openai_client) for _ in range(20))
                for _ in range(10)
            ),
            pbar_desc="gather",
        )
        await model.train(
            train_groups,
            config=art.TrainConfig(learning_rate=1e-4),
            _config=art.dev.TrainConfig(
                precalculate_logprobs=True,
            ),
        )


if __name__ == "__main__":
    asyncio.run(train())
