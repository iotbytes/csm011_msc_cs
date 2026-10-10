document.addEventListener("DOMContentLoaded", () => {
    const input = document.getElementById("imageInput");
    const preview = document.getElementById("preview");
    const button = document.getElementById("analyzeButton");
    const results = document.getElementById("results");

    input.addEventListener("change", () => {
        const file = input.files[0];

        if (!file) {
            button.disabled = true;
            preview.innerHTML = "<p>No image selected</p>";
            return;
        }

        const image = document.createElement("img");
        image.src = URL.createObjectURL(file);
        image.alt = "Selected image";

        preview.innerHTML = "";
        preview.appendChild(image);

        button.disabled = false;
    });

    button.addEventListener("click", async () => {
        const file = input.files[0];

        if (!file) {
            alert("Please select an image first.");
            return;
        }

        button.disabled = true;
        button.textContent = "Processing...";

        const formData = new FormData();
        formData.append("image", file);

        try {
            const response = await fetch("/analyze", {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || "Server processing failed.");
            }

            document.getElementById("originalSize").textContent =
                `${data.width} × ${data.height}`;

            document.getElementById("paddedSize").textContent =
                `${data.padded_width} × ${data.padded_height}`;

            document.getElementById("totalBlocks").textContent =
                data.total_blocks.toLocaleString();

            document.getElementById("grayscaleImage").src =
                data.images.grayscale;

            document.getElementById("dft50Image").src =
                data.images.dft_50;

            document.getElementById("dft25Image").src =
                data.images.dft_25;

            document.getElementById("dct50Image").src =
                data.images.dct_50;

            document.getElementById("dct25Image").src =
                data.images.dct_25;

            document.getElementById("dft50Mse").textContent =
                data.dft_50_mse.toFixed(6);

            document.getElementById("dft25Mse").textContent =
                data.dft_25_mse.toFixed(6);

            document.getElementById("dct50Mse").textContent =
                data.dct_50_mse.toFixed(6);

            document.getElementById("dct25Mse").textContent =
                data.dct_25_mse.toFixed(6);

            results.style.display = "block";

            results.scrollIntoView({
                behavior: "smooth"
            });

        } catch (error) {
            console.error("Analysis error:", error);
            alert("Analysis failed:\n" + error.message);

        } finally {
            button.disabled = false;
            button.textContent = "Analyze Image";
        }
    });
});