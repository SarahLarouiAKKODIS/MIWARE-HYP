#!/usr/bin/env python3

import json
import os
from tkinter import *
from tkinter import ttk, filedialog, messagebox
from tkinter.ttk import Notebook
from clean_src.mineral_processor import MineralProcessor

class ConfigEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Configuration Editor")
        self.root.geometry("1920x1080")
        
        # Current configuration data
        self.config_data = {}
        self.config_file_path = ""
        self.mineral_processor = MineralProcessor()

        # Create notebook (tabs)
        self.notebook = Notebook(root)
        self.notebook.pack(pady=10, expand=True, fill='both')
        
        # Create config tab
        self.config_tab = Frame(self.notebook)
        self.notebook.add(self.config_tab, text='Config')
        
        # Create process tab (empty for now)
        self.process_tab = Frame(self.notebook)
        self.notebook.add(self.process_tab, text='Process')
        
        # Initialize config tab
        self.init_config_tab()
        
        # Initialize process tab with buttons
        self.init_process_tab()
    
    # Config Tab Methods
    def init_config_tab(self):
        """Initialize the config tab with load button and configuration fields"""
        
        # Load button frame
        load_frame = Frame(self.config_tab)
        load_frame.pack(pady=10, fill='x')
        
        Button(load_frame, text="Load Config", command=self.load_config).pack(side=LEFT, padx=10)
        self.file_label = Label(load_frame, text="No file loaded")
        self.file_label.pack(side=LEFT)
        
        # Configuration frame
        self.config_frame = Frame(self.config_tab)
        self.config_frame.pack(pady=10, fill='both', expand=True)
        
        # Save button
        save_frame = Frame(self.config_tab)
        save_frame.pack(pady=10, fill='x')
        Button(save_frame, text="Save Config", command=self.save_config).pack(side=LEFT, padx=10)
    
    def load_config(self):
        """Load configuration from JSON file"""
        
        # Open file dialog
        file_path = filedialog.askopenfilename(
            initialdir="configs",
            title="Select Config File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'r') as f:
                self.config_data = json.load(f)
                self.config_file_path = file_path
                
            # Update UI
            self.file_label.config(text=os.path.basename(file_path))
            self.display_config_fields()

            # Load mineral list for process tab
            self.mineral_dropdown['values'] = self.get_relab_minerals()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load config: {e}")
    
    def display_config_fields(self):
        """Display configuration fields in the UI"""
        
        # Clear existing widgets
        for widget in self.config_frame.winfo_children():
            widget.destroy()
        
        if not self.config_data:
            Label(self.config_frame, text="No configuration loaded").pack(pady=20)
            return
        
        # Create a scrollable frame
        canvas = Canvas(self.config_frame)
        scrollbar = ttk.Scrollbar(self.config_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Configure grid to expand
        scrollable_frame.grid_columnconfigure(1, weight=1)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Display each configuration field
        row = 0
        for key, value in self.config_data.items():
            self.display_field(scrollable_frame, key, value, row)
            row += 1
    
    def update_config(self, k, v):
        """Update the configuration data when a field is changed"""
        self.config_data[k] = v
        print(f"Updated config: {k} = {v}")

    def display_field(self, parent, key, value, row):
        """Display a single configuration field with appropriate widget"""
        
        # Label for the field
        label = Label(parent, text=key + ":", anchor="w")
        label.grid(row=row, column=0, sticky="ew", padx=5, pady=2)
        
        # Determine widget type based on value type
        if isinstance(value, bool):
            var = BooleanVar(value=value)
            checkbox = Checkbutton(parent, variable=var, onvalue=True, offvalue=False, command=lambda k=key, v=var: self.update_config(k, v.get()))
            checkbox.grid(row=row, column=1, sticky="w", padx=5, pady=2)
            
        elif isinstance(value, (int, float)):
            entry_var = StringVar(value=str(value))
            entry = Entry(parent, textvariable=entry_var, width=30)
            entry.grid(row=row, column=1, sticky="w", padx=5, pady=2)
            entry.bind("<FocusOut>", lambda e, k=key, v=entry_var: self.update_config(k, float(v.get()) if '.' in v.get() else int(v.get())))
            
        elif isinstance(value, str):
            # Check if the string looks like a file or directory path
            if self._looks_like_path(value):
                # Create appropriate widget for path selection
                frame = Frame(parent)
                frame.grid(row=row, column=1, sticky="w", padx=5, pady=2)
                
                entry_var = StringVar(value=value)
                entry = Entry(frame, textvariable=entry_var, width=100)
                entry.pack(side=LEFT)
                
                # Add file/directory selection button
                if os.path.isdir(value) or value.endswith(os.sep):
                    # Directory path
                    Button(frame, text="...", command=lambda k=key, v=entry_var: self._select_directory(k, v)).pack(side=LEFT, padx=(5, 0))
                else:
                    # File path
                    Button(frame, text="...", command=lambda k=key, v=entry_var: self._select_file(k, v)).pack(side=LEFT, padx=(5, 0))
            else:
                entry_var = StringVar(value=value)
                entry = Entry(parent, textvariable=entry_var, width=100)
                entry.grid(row=row, column=1, sticky="w", padx=5, pady=2)
                entry.bind("<FocusOut>", lambda e, k=key, v=entry_var: self.update_config(k, v.get()))
            
        elif isinstance(value, dict):
            # For nested dictionaries, create a collapsible section
            var = BooleanVar(value=False)
            checkbox = Checkbutton(parent, text="Expand", variable=var, command=lambda: self.toggle_nested_section(parent, key, value, row))
            checkbox.grid(row=row, column=1, sticky="w", padx=5, pady=2)
            
        elif isinstance(value, list):
            # For lists, show as comma-separated string
            entry_var = StringVar(value=", ".join(map(str, value)))
            entry = Entry(parent, textvariable=entry_var, width=30)
            entry.grid(row=row, column=1, sticky="w", padx=5, pady=2)
            entry.bind("<FocusOut>", lambda e, k=key, v=entry_var: self.update_config(k, [float(item.strip()) if '.' in item.strip() else int(item.strip()) for item in v.get().split(",")]))
        
        # Store reference to update config when changed
        # self.config_data[key] = value
    
    def _looks_like_path(self, text):
        """Check if a string looks like a file or directory path"""
        # Check for common path separators
        if os.sep in text or (os.altsep and os.altsep in text):
            return True
        
        # Check for Windows drive letters
        if ':' in text and text[1] == ':':
            return True
        
        # Check for common path patterns
        if '/' in text or '\\' in text:
            return True
        
        return False
    
    def _select_file(self, key, entry_var):
        """Select a file and update the entry variable"""
        file_path = filedialog.askopenfilename()
        if file_path:
            entry_var.set(file_path)
            self.config_data[key] = file_path
    
    def _select_directory(self, key, entry_var):
        """Select a directory and update the entry variable"""
        dir_path = filedialog.askdirectory()
        if dir_path:
            entry_var.set(dir_path)
            self.config_data[key] = dir_path
    
    def toggle_nested_section(self, parent, key, value, row):
        """Toggle display of nested dictionary section"""
        pass  # Implementation for nested sections would go here
    
    def save_config(self):
        """Save configuration to JSON file"""
        
        if not self.config_file_path:
            messagebox.showwarning("Warning", "No config file loaded")
            return
        
        try:
            with open(self.config_file_path, 'w') as f:
                json.dump(self.config_data, f, indent=4)
            
            messagebox.showinfo("Success", f"Configuration saved to {self.config_file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save config: {e}")
    
    # Process Tab Methods
    def get_relab_minerals(self):
        """Scan RELAB directory and return list of mineral names from subdirectories"""
        # relab_path = os.path.join("Data", "Librairies_spectrales", "RELAB")
        relab_path = self.config_data["spectral_library_dir"]
        minerals = []
        
        if os.path.exists(relab_path):
            for entry in os.listdir(relab_path):
                full_path = os.path.join(relab_path, entry)
                if os.path.isdir(full_path):
                    minerals.append(entry)
        
        return sorted(minerals) if minerals else ["no minerals loaded"]
    
    def init_process_tab(self):
        """Initialize the process tab with processing buttons"""
        
        # Process buttons frame
        button_frame = Frame(self.process_tab)
        button_frame.pack(pady=20, fill='x')
        
        # Mineral selection dropdown
        mineral_frame = Frame(self.process_tab)
        mineral_frame.pack(pady=10, fill='x')
        Label(mineral_frame, text="Mineral:").pack(side=LEFT, padx=5)

        self.selected_mineral = StringVar(value="load config first")
        self.mineral_dropdown = ttk.Combobox(
            mineral_frame,
            textvariable=self.selected_mineral,
            values=["no minerals loaded"],
            state="readonly",
            width=15
        )
        self.mineral_dropdown.pack(side=LEFT, padx=5)

        # Load config button (for reloading in process tab)
        Button(button_frame, text="Load Config", command=self.load_config_from_process).pack(pady=5, fill='x')
        
        # Pre-processing button
        Button(button_frame, text="Pre Processing", command=self.run_preprocessing).pack(pady=5, fill='x')
        
        # Banddepth mineral detection button
        Button(button_frame, text="Banddepth Mineral Detection", command=self.run_banddepth_detection).pack(pady=5, fill='x')
        
        # Spectral comparison mineral detection button
        Button(button_frame, text="Spectral Comparison Mineral Detection", command=self.run_spectral_comparison).pack(pady=5, fill='x')
    
    def load_config_from_process(self):
        """Load configuration from JSON file (called from process tab)"""
        self.load_config()
        self.mineral_processor.load_config_from_path(self.config_file_path)
    
    def run_preprocessing(self):
        """Run the preprocessing pipeline"""
        if not self.config_file_path:
            messagebox.showwarning("Warning", "Please load a configuration file first")
            return
        
        try:
            # Call the mineral processor's process method
            self.mineral_processor.pre_process()
            messagebox.showinfo("Success", "Preprocessing completed successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Preprocessing failed: {e}")
    
    def run_banddepth_detection(self):
        """Run banddepth mineral detection"""
        if not self.config_file_path:
            messagebox.showwarning("Warning", "Please load a configuration file first")
            return
        
        try:
            self.mineral_processor.process_banddepth_mineral_detection()
            messagebox.showinfo("Info", "Banddepth mineral detection would run here")
        except Exception as e:
            messagebox.showerror("Error", f"Banddepth detection failed: {e}")
    
    def run_spectral_comparison(self):
        """Run spectral comparison mineral detection"""
        if not self.config_file_path:
            messagebox.showwarning("Warning", "Please load a configuration file first")
            return
        
        try:
            self.mineral_processor.process_spectral_comparison_mineral_detection(mineral=self.selected_mineral.get())
            messagebox.showinfo("Info", "Spectral comparison mineral detection would run here")
        except Exception as e:
            messagebox.showerror("Error", f"Spectral comparison failed: {e}")

if __name__ == "__main__":
    root = Tk()
    app = ConfigEditor(root)
    root.mainloop()