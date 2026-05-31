import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from PIL import Image, ImageTk
from ultralytics import YOLO
from pathlib import Path
import os
import time
import threading

class ModernStyle:
    """Modern color scheme"""
    PRIMARY = "#1E1E2E"      # Dark background
    SECONDARY = "#2D2D44"    # Slightly lighter
    ACCENT = "#00D9FF"       # Cyan accent
    ACCENT_ALT = "#FF6B9D"   # Pink accent
    SUCCESS = "#4CAF50"      # Green
    WARNING = "#FF9800"      # Orange
    ERROR = "#F44336"        # Red
    TEXT = "#FFFFFF"         # White text
    TEXT_DIM = "#B0B0B0"     # Dim text

class AircraftClassifierGUIEnhanced:
    def __init__(self, root):
        self.root = root
        self.root.title("Aircraft Classifier")
        self.root.geometry("1600x1000")
        self.root.configure(bg=ModernStyle.PRIMARY)

        # Configure modern style
        self._configure_styles()

        # Load model
        model_path = '/mnt/ai_data/aircraft_training_runs/yolo26m_imgsz640_fixed_classes/weights/best.pt'
        if not os.path.exists(model_path):
            messagebox.showerror("Error", f"Model not found at {model_path}")
            return

        self.model = YOLO(model_path)
        self.current_image_path = None
        self.processing = False

        # Create main frame
        main_frame = tk.Frame(root, bg=ModernStyle.PRIMARY)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Header section
        self._create_header(main_frame)

        # Control panel
        self._create_control_panel(main_frame)

        # Main content area (2 columns)
        content_frame = tk.Frame(main_frame, bg=ModernStyle.PRIMARY)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Left side - Prediction
        self._create_prediction_panel(content_frame)

        # Right side - Visualization
        self._create_visualization_panel(content_frame)

        # Bottom info panel
        self._create_info_panel(main_frame)

        # Status bar
        self._create_status_bar(main_frame)

        # Bind keyboard shortcuts
        self._bind_shortcuts()

    def _configure_styles(self):
        """Configure ttk styles for modern look"""
        style = ttk.Style()
        style.theme_use('clam')

        # Configure custom style
        style.configure('TScale', background=ModernStyle.SECONDARY)
        style.configure('TButton', background=ModernStyle.SECONDARY)
        style.configure('TLabel', background=ModernStyle.PRIMARY, foreground=ModernStyle.TEXT)

    def _create_header(self, parent):
        """Create header section"""
        header = tk.Frame(parent, bg=ModernStyle.SECONDARY, height=60)
        header.pack(fill=tk.X, pady=(0, 10))

        title = tk.Label(header, text="AIRCRAFT CLASSIFIER - PROFESSIONAL EDITION",
                        font=("Arial", 16, "bold"), bg=ModernStyle.SECONDARY,
                        fg=ModernStyle.ACCENT)
        title.pack(side=tk.LEFT, padx=20, pady=15)


    def _create_control_panel(self, parent):
        """Create control panel"""
        control_frame = tk.Frame(parent, bg=ModernStyle.SECONDARY, relief=tk.RAISED, bd=1)
        control_frame.pack(fill=tk.X, pady=10)

        # Main button
        self.select_btn = tk.Button(control_frame, text="SELECT IMAGE", command=self.select_image,
                                   font=("Arial", 11, "bold"), bg=ModernStyle.SUCCESS,
                                   fg=ModernStyle.TEXT, padx=20, pady=12, relief=tk.FLAT,
                                   activebackground="#45a049")
        self.select_btn.pack(side=tk.LEFT, padx=15, pady=10)

        # Settings
        settings_frame = tk.Frame(control_frame, bg=ModernStyle.SECONDARY)
        settings_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=15, pady=10)

        # Confidence slider
        tk.Label(settings_frame, text="CONFIDENCE:", font=("Arial", 9, "bold"),
                bg=ModernStyle.SECONDARY, fg=ModernStyle.ACCENT).pack(side=tk.LEFT, padx=5)
        self.conf_slider = tk.Scale(settings_frame, from_=0.0, to=1.0, resolution=0.05,
                                   orient=tk.HORIZONTAL, length=150, bg=ModernStyle.SECONDARY,
                                   fg=ModernStyle.ACCENT, troughcolor=ModernStyle.PRIMARY,
                                   highlightthickness=0)
        self.conf_slider.set(0.25)
        self.conf_slider.pack(side=tk.LEFT, padx=5)

        # Layer selector
        tk.Label(settings_frame, text="LAYER:", font=("Arial", 9, "bold"),
                bg=ModernStyle.SECONDARY, fg=ModernStyle.ACCENT).pack(side=tk.LEFT, padx=5)
        self.layer_var = tk.StringVar(value="stage0")
        layers = [f"stage{i}" for i in range(23)]
        layer_dropdown = tk.OptionMenu(settings_frame, self.layer_var, *layers)
        layer_dropdown.config(bg=ModernStyle.SECONDARY, fg=ModernStyle.TEXT,
                             activebackground=ModernStyle.ACCENT, width=10)
        layer_dropdown["menu"].config(bg=ModernStyle.SECONDARY, fg=ModernStyle.TEXT,
                                     activebackground=ModernStyle.ACCENT)
        layer_dropdown.pack(side=tk.LEFT, padx=5)

        # Action buttons
        self.reprocess_btn = tk.Button(settings_frame, text="REPROCESS",
                                      command=self.reprocess_image,
                                      font=("Arial", 9, "bold"), bg=ModernStyle.ACCENT_ALT,
                                      fg=ModernStyle.TEXT, padx=10, pady=5, relief=tk.FLAT)
        self.reprocess_btn.pack(side=tk.LEFT, padx=5)

        # Right side buttons
        right_buttons = tk.Frame(control_frame, bg=ModernStyle.SECONDARY)
        right_buttons.pack(side=tk.RIGHT, padx=15, pady=10)

        self.batch_btn = tk.Button(right_buttons, text="BATCH", command=self.batch_process,
                                  font=("Arial", 9, "bold"), bg=ModernStyle.WARNING,
                                  fg=ModernStyle.TEXT, padx=10, relief=tk.FLAT)
        self.batch_btn.pack(side=tk.LEFT, padx=5)

        self.save_btn = tk.Button(right_buttons, text="SAVE", command=self.save_results,
                                 font=("Arial", 9, "bold"), bg="#9C27B0",
                                 fg=ModernStyle.TEXT, padx=10, relief=tk.FLAT)
        self.save_btn.pack(side=tk.LEFT, padx=5)

    def _create_prediction_panel(self, parent):
        """Create prediction display panel"""
        left_frame = tk.LabelFrame(parent, text="YOLO DETECTION & CLASSIFICATION",
                                   font=("Arial", 11, "bold"), bg=ModernStyle.SECONDARY,
                                   fg=ModernStyle.ACCENT, relief=tk.RAISED, bd=2)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.pred_canvas = tk.Canvas(left_frame, bg=ModernStyle.PRIMARY,
                                     width=650, height=600, relief=tk.SUNKEN, bd=2)
        self.pred_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.pred_label = tk.Label(left_frame, text="Waiting for image...",
                                  font=("Arial", 9), bg=ModernStyle.SECONDARY,
                                  fg=ModernStyle.TEXT_DIM)
        self.pred_label.pack(pady=5)

    def _create_visualization_panel(self, parent):
        """Create visualization display panel"""
        right_frame = tk.LabelFrame(parent, text="NEURAL NETWORK VISUALIZATION",
                                    font=("Arial", 11, "bold"), bg=ModernStyle.SECONDARY,
                                    fg=ModernStyle.ACCENT, relief=tk.RAISED, bd=2)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        self.viz_canvas = tk.Canvas(right_frame, bg=ModernStyle.PRIMARY,
                                    width=650, height=600, relief=tk.SUNKEN, bd=2)
        self.viz_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.viz_label = tk.Label(right_frame, text="Waiting for image...",
                                 font=("Arial", 9), bg=ModernStyle.SECONDARY,
                                 fg=ModernStyle.TEXT_DIM)
        self.viz_label.pack(pady=5)

    def _create_info_panel(self, parent):
        """Create detection info panel"""
        info_frame = tk.LabelFrame(parent, text="DETECTION RESULTS",
                                  font=("Arial", 10, "bold"), bg=ModernStyle.SECONDARY,
                                  fg=ModernStyle.ACCENT, relief=tk.RAISED, bd=2)
        info_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=(10, 0))

        self.info_text = scrolledtext.ScrolledText(info_frame, height=3,
                                                   font=("Courier", 9),
                                                   bg=ModernStyle.PRIMARY,
                                                   fg=ModernStyle.ACCENT,
                                                   insertbackground=ModernStyle.ACCENT)
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.info_text.config(state=tk.DISABLED)

        # Progress bar
        progress_frame = tk.Frame(info_frame, bg=ModernStyle.SECONDARY)
        progress_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var,
                                           maximum=100, length=300, mode='determinate')
        self.progress_bar.pack(fill=tk.X, expand=True)

    def _create_status_bar(self, parent):
        """Create status bar at bottom"""
        status_frame = tk.Frame(parent, bg=ModernStyle.SECONDARY, relief=tk.SUNKEN, bd=1)
        status_frame.pack(fill=tk.X, pady=(10, 0))

        self.status_label = tk.Label(status_frame, text="Ready | Press Ctrl+O to open image or Ctrl+S to save",
                                    font=("Arial", 9), bg=ModernStyle.SECONDARY,
                                    fg=ModernStyle.TEXT_DIM, justify=tk.LEFT)
        self.status_label.pack(side=tk.LEFT, padx=10, pady=8)

        self.device_label = tk.Label(status_frame, text="", font=("Arial", 9),
                                    bg=ModernStyle.SECONDARY, fg=ModernStyle.SUCCESS)
        self.device_label.pack(side=tk.RIGHT, padx=10, pady=8)

        # Show device info
        import torch
        if torch.cuda.is_available():
            device_info = f"{torch.cuda.get_device_name(0)}"
        else:
            device_info = "CPU Mode"
        self.device_label.config(text=device_info)

    def _bind_shortcuts(self):
        """Bind keyboard shortcuts"""
        self.root.bind('<Control-o>', lambda e: self.select_image())
        self.root.bind('<Control-s>', lambda e: self.save_results())
        self.root.bind('<Control-r>', lambda e: self.reprocess_image())

    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Select aircraft image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if file_path:
            self.current_image_path = file_path
            self.process_image_async(file_path)

    def process_image_async(self, image_path):
        """Process image in background thread"""
        if self.processing:
            messagebox.showwarning("Warning", "Processing already in progress")
            return

        self.processing = True
        self.select_btn.config(state=tk.DISABLED)
        self.progress_var.set(0)

        thread = threading.Thread(target=self._process_worker, args=(image_path,), daemon=True)
        thread.start()

    def _process_worker(self, image_path):
        """Worker thread for image processing"""
        try:
            filename = Path(image_path).name
            self.status_label.config(text=f"Processing: {filename}...")
            self.root.update()

            start_time = time.time()
            self.progress_var.set(20)
            self.pred_label.config(text=f"{filename}...", fg=ModernStyle.TEXT_DIM)

            conf = self.conf_slider.get()
            self.progress_var.set(30)

            results = self.model.predict(
                source=image_path,
                visualize=True,
                save=True,
                conf=conf,
                name="gui_predict",
                exist_ok=True,
                verbose=False
            )

            self.progress_var.set(60)
            inference_time = time.time() - start_time

            # Get prediction image
            pred_image_path = self._get_prediction_image_from_results(results)
            if pred_image_path and os.path.exists(pred_image_path):
                self.display_image_on_canvas(pred_image_path, self.pred_canvas)
                self.pred_label.config(text=f"{filename}", fg=ModernStyle.SUCCESS)
            else:
                self.pred_label.config(text="Prediction not found", fg=ModernStyle.WARNING)

            self.progress_var.set(75)

            # Get visualization
            selected_layer = self.layer_var.get()
            viz_image_path = self._get_visualization_image(selected_layer)
            if viz_image_path and os.path.exists(viz_image_path):
                self.display_image_on_canvas(viz_image_path, self.viz_canvas, max_height=600)
                self.viz_label.config(text=f"{selected_layer}", fg=ModernStyle.SUCCESS)
            else:
                self.viz_label.config(text=f"{selected_layer} not found", fg=ModernStyle.WARNING)

            self.progress_var.set(90)
            self._update_info_panel(results, inference_time)

            self.progress_var.set(100)
            self.status_label.config(text=f"Processed: {filename} ({inference_time:.2f}s)")

        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)[:50]}")
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
        finally:
            self.processing = False
            self.select_btn.config(state=tk.NORMAL)

    def reprocess_image(self):
        """Reprocess with new settings"""
        if self.current_image_path:
            self.process_image_async(self.current_image_path)
        else:
            messagebox.showwarning("Warning", "No image loaded")

    def batch_process(self):
        """Batch process images"""
        folder_path = filedialog.askdirectory(title="Select folder")
        if not folder_path:
            return

        image_files = list(Path(folder_path).glob("*.jpg")) + \
                     list(Path(folder_path).glob("*.jpeg")) + \
                     list(Path(folder_path).glob("*.png"))

        if not image_files:
            messagebox.showwarning("Warning", "No images found")
            return

        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert(tk.END, f"Batch processing {len(image_files)} images...\n")
        self.root.update()

        results_summary = []
        conf = self.conf_slider.get()

        for i, img_file in enumerate(image_files, 1):
            self.progress_var.set((i / len(image_files)) * 100)
            self.info_text.insert(tk.END, f"[{i}/{len(image_files)}] {img_file.name}...")
            self.root.update()

            try:
                result = self.model.predict(source=str(img_file), conf=conf, verbose=False)
                if result and result[0].boxes:
                    detections = []
                    for box, conf_score, cls_id in zip(result[0].boxes.xyxy, result[0].boxes.conf, result[0].boxes.cls):
                        class_name = self.model.names[int(cls_id)]
                        detections.append(f"{class_name}:{float(conf_score):.2f}")
                    results_summary.append((img_file.name, ", ".join(detections)))
                    self.info_text.insert(tk.END, " OK\n")
                else:
                    self.info_text.insert(tk.END, " (no detections)\n")
            except Exception as e:
                self.info_text.insert(tk.END, f" ERROR\n")

            self.root.update()

        self.progress_var.set(100)
        self.info_text.insert(tk.END, f"\n{'='*50}\nResults:\n")
        for filename, detections in results_summary:
            self.info_text.insert(tk.END, f"{filename}: {detections}\n")

        self.info_text.config(state=tk.DISABLED)
        self.status_label.config(text=f"Batch complete: {len(results_summary)}/{len(image_files)} had detections")
        messagebox.showinfo("Success", f"Batch processing complete!")

    def save_results(self):
        """Save results"""
        if not self.current_image_path:
            messagebox.showwarning("Warning", "No image loaded")
            return

        save_dir = filedialog.askdirectory(title="Save to folder")
        if not save_dir:
            return

        try:
            import shutil
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_path = Path(save_dir) / f"aircraft_detection_{timestamp}"
            save_path.mkdir(exist_ok=True)

            shutil.copy(self.current_image_path, save_path / "input.jpg")

            pred_img = self._get_latest_prediction_image()
            if pred_img and os.path.exists(pred_img):
                shutil.copy(pred_img, save_path / "prediction.jpg")

            viz_img = self._get_visualization_image()
            if viz_img and os.path.exists(viz_img):
                shutil.copy(viz_img, save_path / "visualization.png")

            self.status_label.config(text=f"Saved to: {save_path}")
            messagebox.showinfo("Success", f"Results saved!")
        except Exception as e:
            messagebox.showerror("Error", f"Save failed: {str(e)}")

    def _get_prediction_image_from_results(self, results):
        """Extract prediction image for current image"""
        try:
            if not self.current_image_path:
                return None

            # Get current image filename
            current_image_name = Path(self.current_image_path).name

            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'save_dir'):
                    save_dir = Path(result.save_dir)
                    # Look for the exact image file
                    target_img = save_dir / current_image_name
                    if target_img.exists():
                        return str(target_img)

                    # Fallback: get most recent JPG in save_dir
                    jpg_files = sorted([f for f in save_dir.glob("*.jpg") if f.is_file()])
                    if jpg_files:
                        # Return the one matching our current image name
                        for jpg in jpg_files:
                            if current_image_name in jpg.name:
                                return str(jpg)
                        # If no exact match, return the last one
                        return str(jpg_files[-1])
        except Exception as e:
            print(f"Error getting prediction from results: {e}")

        return self._get_latest_prediction_image()

    def _get_latest_prediction_image(self):
        """Get latest prediction for current image"""
        if not self.current_image_path:
            return None

        current_image_name = Path(self.current_image_path).name  # filename with extension

        search_paths = [Path("runs/detect/runs/detect"), Path("runs/detect")]

        for base_path in search_paths:
            if not base_path.exists():
                continue

            predict_dirs = sorted([d for d in base_path.iterdir()
                                 if d.is_dir() and "gui_predict" in d.name], reverse=True)

            if predict_dirs:
                # Check last 3 prediction runs for the current image
                for predict_dir in predict_dirs[:3]:
                    # Look for prediction file matching current image
                    jpg_file = predict_dir / current_image_name
                    if jpg_file.exists():
                        return str(jpg_file)

        return None

    def _get_visualization_image(self, layer=None):
        """Get visualization for current image and layer"""
        # Get the current image filename without extension
        if not self.current_image_path:
            return None

        current_image_name = Path(self.current_image_path).stem  # filename without extension

        search_paths = [Path("runs/detect/runs/detect"), Path("runs/detect")]

        for base_path in search_paths:
            if not base_path.exists():
                continue

            predict_dirs = sorted([d for d in base_path.iterdir()
                                 if d.is_dir() and "gui_predict" in d.name], reverse=True)

            if predict_dirs:
                # Look for files from the current image
                for predict_dir in predict_dirs[:3]:  # Check last 3 prediction runs
                    # Find subdirectory matching current image
                    image_subdir = predict_dir / current_image_name
                    if image_subdir.exists():
                        png_files = sorted(image_subdir.glob("*.png"))

                        if layer:
                            layer_files = [f for f in png_files if layer in f.name]
                            if layer_files:
                                return str(layer_files[0])
                        else:
                            if png_files:
                                return str(png_files[0])

        return None

    def _update_info_panel(self, results, inference_time):
        """Update info"""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete("1.0", tk.END)

        info = f"Inference: {inference_time*1000:.1f}ms | Confidence: {self.conf_slider.get():.2f}\n"
        info += "─" * 60 + "\n"

        if results and len(results) > 0 and results[0].boxes:
            info += f"Detections: {len(results[0].boxes)}\n"
            for i, (conf_score, cls_id) in enumerate(zip(results[0].boxes.conf, results[0].boxes.cls), 1):
                class_name = self.model.names[int(cls_id)]
                info += f"  {i}. {class_name}: {float(conf_score):.3f}\n"
        else:
            info += "No detections at current threshold\n"

        self.info_text.insert(tk.END, info)
        self.info_text.config(state=tk.DISABLED)

    def display_image_on_canvas(self, image_path, canvas, max_height=600):
        """Display image"""
        try:
            img = Image.open(image_path)
            canvas_width = canvas.winfo_width()
            if canvas_width <= 1:
                canvas_width = 620

            img.thumbnail((canvas_width - 20, max_height - 20), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)

            canvas.delete("all")
            canvas.create_image(canvas_width // 2, max_height // 2, image=photo)

            if not hasattr(canvas, '_photo_images'):
                canvas._photo_images = []
            canvas._photo_images.append(photo)
            if len(canvas._photo_images) > 3:
                canvas._photo_images.pop(0)
        except Exception as e:
            print(f"Error: {e}")

def main():
    root = tk.Tk()
    gui = AircraftClassifierGUIEnhanced(root)
    root.mainloop()

if __name__ == "__main__":
    main()

