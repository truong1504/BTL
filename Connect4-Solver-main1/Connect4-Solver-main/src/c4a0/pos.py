from enum import Enum
from typing import Optional, Tuple
import numpy as np

class CellValue(Enum):
    OPPONENT = 0
    PLAYER = 1

class TerminalState(Enum):
    PLAYER_WIN = 1
    OPPONENT_WIN = 2
    DRAW = 3

class Pos:
    N_ROWS = 6
    N_COLS = 7

    # Tạo danh sách các bitmask chiến thắng (WIN_MASKS) khi định nghĩa lớp
    WIN_MASKS = []
    # Ngang
    for row in range(N_ROWS):
        for col in range(N_COLS - 3):
            mask = (1 << (row * N_COLS + col)) | \
                   (1 << (row * N_COLS + col + 1)) | \
                   (1 << (row * N_COLS + col + 2)) | \
                   (1 << (row * N_COLS + col + 3))
            WIN_MASKS.append(mask)
    # Dọc
    for col in range(N_COLS):
        for row in range(N_ROWS - 3):
            mask = (1 << (row * N_COLS + col)) | \
                   (1 << ((row + 1) * N_COLS + col)) | \
                   (1 << ((row + 2) * N_COLS + col)) | \
                   (1 << ((row + 3) * N_COLS + col))
            WIN_MASKS.append(mask)
    # Chéo từ trên-trái xuống dưới-phải
    for row in range(N_ROWS - 3):
        for col in range(N_COLS - 3):
            mask = (1 << (row * N_COLS + col)) | \
                   (1 << ((row + 1) * N_COLS + col + 1)) | \
                   (1 << ((row + 2) * N_COLS + col + 2)) | \
                   (1 << ((row + 3) * N_COLS + col + 3))
            WIN_MASKS.append(mask)
    # Chéo từ dưới-trái lên trên-phải
    for row in range(3, N_ROWS):
        for col in range(N_COLS - 3):
            mask = (1 << (row * N_COLS + col)) | \
                   (1 << ((row - 1) * N_COLS + col + 1)) | \
                   (1 << ((row - 2) * N_COLS + col + 2)) | \
                   (1 << ((row - 3) * N_COLS + col + 3))
            WIN_MASKS.append(mask)

    def __init__(self, mask=0, value=0):
        """Khởi tạo trạng thái với mask (vị trí có quân) và value (màu quân của người chơi hiện tại)."""
        self.mask = mask
        self.value = value

    @staticmethod
    def _idx_mask_unsafe(row, col):
        """Tính bitmask cho ô tại (row, col) mà không kiểm tra giới hạn."""
        idx = row * Pos.N_COLS + col
        return 1 << idx

    @staticmethod
    def from_2d_board(board: list[list[int]]) -> 'Pos':
        """
        Chuyển đổi bàn cờ 2D thành đối tượng Pos.
        Input: board là List[List[int]] kích thước 6x7, với 0 = trống, 1 = người chơi 1, 2 = người chơi 2.
        Output: Đối tượng Pos với mask và value biểu diễn trạng thái bàn cờ.
        """
        mask = 0
        value = 0

        # Đếm số ô có quân để xác định lượt chơi
        tmp = sum(1 for row in board for cell in row if cell != 0)
        current_turn_value = 1 if tmp % 2 == 0 else 2

        # Duyệt qua bàn cờ để xây dựng mask và value
        for user_row in range(Pos.N_ROWS):
            pos_row = Pos.N_ROWS - 1 - user_row  # Chuyển đổi từ hàng trên xuống sang hàng dưới lên
            for col in range(Pos.N_COLS):
                cell = board[user_row][col]
                if cell != 0:
                    idx = Pos._idx_mask_unsafe(pos_row, col)
                    mask |= idx  # Đặt bit trong mask cho mọi ô có quân
                    if cell == current_turn_value:
                        value |= idx  # Đặt bit trong value nếu ô thuộc người chơi hiện tại

        # In thông tin để kiểm tra (tương tự mã Rust)
        print(f"mask: {mask:064b}")
        print(f"value: {value:064b}")
        print(f"current_turn_value: {current_turn_value}")

        return Pos(mask, value)

    def get(self, row, col):
        """Lấy giá trị ô tại (row, col): None nếu trống, hoặc CellValue."""
        if row >= self.N_ROWS or col >= self.N_COLS:
            return None
        idx = self._idx_mask_unsafe(row, col)
        if (self.mask & idx) == 0:
            return None
        return CellValue.PLAYER if (self.value & idx) != 0 else CellValue.OPPONENT

    def _set_piece_unsafe(self, row, col, piece):
        """Tạo trạng thái mới với quân cờ tại (row, col) mà không kiểm tra."""
        idx = self._idx_mask_unsafe(row, col)
        if piece is None:
            new_mask = self.mask & ~idx
            new_value = self.value & ~idx
        else:
            new_mask = self.mask | idx
            new_value = self.value | idx if piece == CellValue.PLAYER else self.value & ~idx
        return Pos(new_mask, new_value)

    def invert(self):
        """Đảo màu quân cờ, trả về trạng thái mới từ góc nhìn đối thủ."""
        new_value = ~self.value & self.mask
        return Pos(self.mask, new_value)

    def make_move(self, col):
        """Thực hiện nước đi tại cột col, trả về trạng thái mới hoặc None nếu không hợp lệ."""
        if col >= self.N_COLS:
            return None
        for row in range(self.N_ROWS):
            if self.get(row, col) is None:
                new_pos = self._set_piece_unsafe(row, col, CellValue.PLAYER)
                return new_pos.invert()
        return None

    def ply(self):
        """Trả về số nước đi đã chơi (số ô đã chiếm)."""
        return bin(self.mask).count('1')

    def _is_terminal_for_player(self):
        """Kiểm tra xem người chơi hiện tại có thắng không."""
        player_tokens = self.mask & self.value
        for win_mask in self.WIN_MASKS:
            if bin(player_tokens & win_mask).count('1') == 4:
                return True
        return False

    def is_terminal_state(self):
        """Kiểm tra trạng thái kết thúc: thắng, thua, hòa, hoặc None nếu chưa kết thúc."""
        if self._is_terminal_for_player():
            return TerminalState.PLAYER_WIN
        inverted = self.invert()
        if inverted._is_terminal_for_player():
            return TerminalState.OPPONENT_WIN
        if self.ply() == self.N_ROWS * self.N_COLS:
            return TerminalState.DRAW
        return None

    def legal_moves(self):
        """Trả về danh sách boolean chỉ ra các cột có thể chơi."""
        top_row = self.N_ROWS - 1
        return [self.get(top_row, col) is None for col in range(self.N_COLS)]

    def get_terminal_value(self):
        """Trả về giá trị kết thúc: 1.0 (thắng), -1.0 (thua), 0.0 (hòa), hoặc None."""
        state = self.is_terminal_state()
        if state == TerminalState.PLAYER_WIN:
            return 1.0
        elif state == TerminalState.OPPONENT_WIN:
            return -1.0
        elif state == TerminalState.DRAW:
            return 0.0
        return None
    def terminal_value_with_ply_penalty(self, c_ply_penalty: float) -> Optional[Tuple[float, float]]:
        state = self.is_terminal_state()
        if state is None:
            return None
        ply_penalty_magnitude = c_ply_penalty * self.ply()
        if state == TerminalState.PLAYER_WIN:
            return (1.0 - ply_penalty_magnitude, 1.0)
        elif state == TerminalState.OPPONENT_WIN:
            return (-1.0 + ply_penalty_magnitude, -1.0)
        elif state == TerminalState.DRAW:
            return (0.0, 0.0)
        else:
            raise ValueError("Invalid terminal state")

    def write_numpy_buffer(self) -> np.ndarray:
        buf = np.zeros((1, 2, self.N_ROWS, self.N_COLS), dtype=np.float32)
        for row in range(self.N_ROWS):
            for col in range(self.N_COLS):
                piece = self.get(row, col)
                if piece == CellValue.PLAYER:
                    buf[0, 0, row, col] = 1.0
                elif piece == CellValue.OPPONENT:
                    buf[0, 1, row, col] = 1.0
        return buf
    