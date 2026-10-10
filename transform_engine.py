import cv2
import numpy as np
import os


def retain_coefficients(coefficients, percentage):
    total = coefficients.size
    count = int(total * percentage)

    magnitude = np.abs(coefficients)
    indices = np.argsort(magnitude.ravel())[::-1]

    filtered = np.zeros_like(coefficients)

    selected = indices[:count]
    filtered.ravel()[selected] = coefficients.ravel()[selected]

    return filtered


def process_image(input_path, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    image = cv2.imread(input_path)

    if image is None:
        raise ValueError("Image could not be loaded.")

    # Grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    height, width = gray.shape

    # Padding
    pad_height = (8 - height % 8) % 8
    pad_width = (8 - width % 8) % 8

    padded = np.pad(
        gray,
        ((0, pad_height), (0, pad_width)),
        mode="constant"
    )

    padded_height, padded_width = padded.shape

    vertical_blocks = padded_height // 8
    horizontal_blocks = padded_width // 8

    total_blocks = (
        vertical_blocks * horizontal_blocks
    )

    # Output images
    dft_50 = np.zeros_like(
        padded,
        dtype=np.float64
    )

    dft_25 = np.zeros_like(
        padded,
        dtype=np.float64
    )

    dct_50 = np.zeros_like(
        padded,
        dtype=np.float64
    )

    dct_25 = np.zeros_like(
        padded,
        dtype=np.float64
    )

    # Process 8x8 blocks
    for row in range(vertical_blocks):

        for col in range(horizontal_blocks):

            y = row * 8
            x = col * 8

            block = padded[
                y:y + 8,
                x:x + 8
            ].astype(np.float64)

            # DFT
            dft = np.fft.fft2(block)

            dft50 = retain_coefficients(
                dft,
                0.50
            )

            dft25 = retain_coefficients(
                dft,
                0.25
            )

            dft_50[
                y:y + 8,
                x:x + 8
            ] = np.real(
                np.fft.ifft2(dft50)
            )

            dft_25[
                y:y + 8,
                x:x + 8
            ] = np.real(
                np.fft.ifft2(dft25)
            )

            # DCT
            dct = cv2.dct(block)

            dct50 = retain_coefficients(
                dct,
                0.50
            )

            dct25 = retain_coefficients(
                dct,
                0.25
            )

            dct_50[
                y:y + 8,
                x:x + 8
            ] = cv2.idct(dct50)

            dct_25[
                y:y + 8,
                x:x + 8
            ] = cv2.idct(dct25)

    # Remove padding
    dft_50 = dft_50[:height, :width]
    dft_25 = dft_25[:height, :width]

    dct_50 = dct_50[:height, :width]
    dct_25 = dct_25[:height, :width]

    # Convert to image format
    dft_50 = np.uint8(
        np.clip(
            np.round(dft_50),
            0,
            255
        )
    )

    dft_25 = np.uint8(
        np.clip(
            np.round(dft_25),
            0,
            255
        )
    )

    dct_50 = np.uint8(
        np.clip(
            np.round(dct_50),
            0,
            255
        )
    )

    dct_25 = np.uint8(
        np.clip(
            np.round(dct_25),
            0,
            255
        )
    )

    # MSE
    original = gray.astype(np.float64)

    mse_dft_50 = np.mean(
        (original - dft_50) ** 2
    )

    mse_dft_25 = np.mean(
        (original - dft_25) ** 2
    )

    mse_dct_50 = np.mean(
        (original - dct_50) ** 2
    )

    mse_dct_25 = np.mean(
        (original - dct_25) ** 2
    )

    # Error images
    error_dft_50 = cv2.absdiff(
        gray,
        dft_50
    )

    error_dft_25 = cv2.absdiff(
        gray,
        dft_25
    )

    error_dct_50 = cv2.absdiff(
        gray,
        dct_50
    )

    error_dct_25 = cv2.absdiff(
        gray,
        dct_25
    )

    # Save images
    cv2.imwrite(
        os.path.join(
            output_dir,
            "grayscale.png"
        ),
        gray
    )

    cv2.imwrite(
        os.path.join(
            output_dir,
            "dft_50.png"
        ),
        dft_50
    )

    cv2.imwrite(
        os.path.join(
            output_dir,
            "dft_25.png"
        ),
        dft_25
    )

    cv2.imwrite(
        os.path.join(
            output_dir,
            "dct_50.png"
        ),
        dct_50
    )

    cv2.imwrite(
        os.path.join(
            output_dir,
            "dct_25.png"
        ),
        dct_25
    )

    cv2.imwrite(
        os.path.join(
            output_dir,
            "dft_50_error.png"
        ),
        error_dft_50
    )

    cv2.imwrite(
        os.path.join(
            output_dir,
            "dft_25_error.png"
        ),
        error_dft_25
    )

    cv2.imwrite(
        os.path.join(
            output_dir,
            "dct_50_error.png"
        ),
        error_dct_50
    )

    cv2.imwrite(
        os.path.join(
            output_dir,
            "dct_25_error.png"
        ),
        error_dct_25
    )

    return {
        "width": width,
        "height": height,

        "padded_width": padded_width,
        "padded_height": padded_height,

        "pad_width": pad_width,
        "pad_height": pad_height,

        "vertical_blocks": vertical_blocks,
        "horizontal_blocks": horizontal_blocks,
        "total_blocks": total_blocks,

        "dft_50_mse": float(mse_dft_50),
        "dft_25_mse": float(mse_dft_25),

        "dct_50_mse": float(mse_dct_50),
        "dct_25_mse": float(mse_dct_25)
    }
