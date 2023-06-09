import pygame
# 画面サイズと背景色を設定
WIDTH, HEIGHT = 900, 900
BACKGROUND_COLOR = (255, 255, 255)  # 白色
# 初期化
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
# 画面を背景色で塗りつぶす
screen.fill(BACKGROUND_COLOR)
pygame.display.set_caption("Drawing")
clock = pygame.time.Clock()
# 描画用の変数
drawing = False
last_pos = None
# メインループ
running = True
while running:
    # イベント処理
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # 左ボタンをクリックしたら描画開始
                drawing = True
                last_pos = event.pos
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:  # 左ボタンを離したら描画終了
                drawing = False
        elif event.type == pygame.MOUSEMOTION:
            if drawing:
                current_pos = event.pos
                if last_pos is not None:
                    pygame.draw.line(screen, (0, 0, 0), last_pos, current_pos, 50)
                last_pos = current_pos

    # 画面の更新
    pygame.display.flip()
    clock.tick(60)  # FPSを60に設定

# 画像を保存

pygame.image.save(screen, "drawn_image.png")

# 終了処理
pygame.quit()