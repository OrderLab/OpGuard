#!/usr/bin/env python3
"""
Test program that directly uses the callsite tracer.
This demonstrates the proper way to use the OpGuard callsite tracer.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import random
import json
import os
import sys
from pathlib import Path
import glob
import re

# Import the callsite tracer
from opguard.callsite_tracer.trace import CallsiteTracer

class SimpleNet(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def get_file_with_largest_suffix(prefix="callsites_traced_rank", suffix=".json"):
    max_index = -1
    selected_file = None

    pattern = re.compile(rf"{re.escape(prefix)}\d+_(\d+){re.escape(suffix)}")

    for file in glob.glob(f"{prefix}*{suffix}"):
        match = pattern.fullmatch(file)
        if match:
            index = int(match.group(1))
            if index > max_index:
                max_index = index
                selected_file = file

    return selected_file

def create_dummy_data(batch_size=32, input_size=10):
    """Create dummy training data."""
    if hasattr(torch, 'cuda') and torch.cuda.is_available():
        X = torch.randn(batch_size, input_size).cuda()
        y = torch.randint(0, 5, (batch_size,)).cuda()
    else:
        X = torch.randn(batch_size, input_size)
        y = torch.randint(0, 5, (batch_size,))
    return X, y

def training_step(model, optimizer, criterion, X, y):
    """Single training step with explicit tensor operations."""
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # Move tensors to GPU for all operations
    if hasattr(torch, 'cuda') and torch.cuda.is_available():
        # Move tensors to GPU
        X_gpu = X.cuda()
        outputs_gpu = outputs.cuda()
        
        # GPU tensor operations
        temp_gpu = X_gpu @ X_gpu.T
        temp_gpu = torch.relu(temp_gpu)
        temp_gpu = torch.sigmoid(temp_gpu)
        temp_gpu = torch.tanh(temp_gpu)
        
        # GPU matrix operations
        temp_gpu = torch.mm(temp_gpu, temp_gpu)
        temp_gpu = torch.add(temp_gpu, temp_gpu)
        temp_gpu = torch.mul(temp_gpu, 0.5)
        
        # GPU element-wise operations
        temp_gpu = torch.exp(temp_gpu)
        temp_gpu = torch.log(temp_gpu + 1e-8)
        temp_gpu = torch.sqrt(temp_gpu)
        
        # Clean up
        del temp_gpu, X_gpu, outputs_gpu
    else:
        # Fallback to CPU if CUDA not available
        temp_cpu = X @ X.T
        temp_cpu = torch.relu(temp_cpu)
        temp_cpu = torch.sigmoid(temp_cpu)
        temp_cpu = torch.tanh(temp_cpu)
        
        # Matrix operations
        temp_cpu = torch.mm(temp_cpu, temp_cpu)
        temp_cpu = torch.add(temp_cpu, temp_cpu)
        temp_cpu = torch.mul(temp_cpu, 0.5)
        
        # Element-wise operations
        temp_cpu = torch.exp(temp_cpu)
        temp_cpu = torch.log(temp_cpu + 1e-8)
        temp_cpu = torch.sqrt(temp_cpu)
        
        # Clean up
        del temp_cpu
    
    # Backward pass
    loss.backward()
    optimizer.step()
    return loss.item()

def validation_step(model, criterion, X, y):
    """Single validation step."""
    with torch.no_grad():
        outputs = model(X)
        loss = criterion(outputs, y)
        
        # Add some operations during validation too
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            temp = torch.randn_like(X).cuda()
            temp = torch.relu(temp)
            temp = torch.sigmoid(temp)
            del temp
        else:
            temp = torch.randn_like(X)
            temp = torch.relu(temp)
            temp = torch.sigmoid(temp)
            del temp
        
    return loss.item()

def main():
    print("Starting SimpleNet training with callsite tracing...")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    
    # Set up output path
    output_path = "callsites_traced.json"
    
    # Create tracer
    tracer = CallsiteTracer(
        dump_every=100,  # Dump more frequently for testing
        dump_path=output_path
    )
    
    # Run the training with tracing
    with tracer:
        # Model setup
        model = SimpleNet()
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            model = model.cuda()
        criterion = nn.CrossEntropyLoss()
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            criterion = criterion.cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        num_epochs = 3
        batch_size = 16
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Training phase
            model.train()
            train_loss = 0.0
            for step in range(5):  # 5 training steps per epoch
                X, y = create_dummy_data(batch_size)
                loss = training_step(model, optimizer, criterion, X, y)
                train_loss += loss
                
                # Additional tensor operations between steps
                if step % 2 == 0:
                    # Create some tensors and perform operations
                    temp_tensor = torch.randn(10, 10)
                    temp_tensor = temp_tensor @ temp_tensor.T
                    temp_tensor = torch.relu(temp_tensor)
                    temp_tensor = torch.sigmoid(temp_tensor)
                    
                    # More complex operations
                    temp_tensor = torch.mm(temp_tensor, temp_tensor)
                    temp_tensor = torch.add(temp_tensor, temp_tensor)
                    temp_tensor = torch.mul(temp_tensor, 0.5)
                    
                    # Element-wise operations
                    temp_tensor = torch.exp(temp_tensor)
                    temp_tensor = torch.log(temp_tensor + 1e-8)
                    temp_tensor = torch.sqrt(temp_tensor)
                    
                    del temp_tensor
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            for step in range(2):  # 2 validation steps per epoch
                X, y = create_dummy_data(batch_size)
                loss = validation_step(model, criterion, X, y)
                val_loss += loss
            
            avg_train_loss = train_loss / 5
            avg_val_loss = val_loss / 2
            print(f"  Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        print("Training completed!")
        
        # Final tensor operations
        print("Performing final tensor operations...")
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            final_tensor = torch.randn(20, 20).cuda()
        else:
            final_tensor = torch.randn(20, 20)
        
        final_tensor = torch.relu(final_tensor)
        final_tensor = torch.sigmoid(final_tensor)
        final_tensor = torch.mm(final_tensor, final_tensor.T)
        final_tensor = torch.add(final_tensor, final_tensor)
        final_tensor = torch.mul(final_tensor, 0.5)
        del final_tensor
        
        # Save a small checkpoint
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': num_epochs,
        }, 'simple_model_checkpoint.pt')
        
        print("Model checkpoint saved!")
    
    output_path = get_file_with_largest_suffix()

    # ============================================================================
    # COMPLETE OPGUARD PIPELINE TEST
    # ============================================================================
    
    print("\n" + "="*60)
    print("OPGUARD END-TO-END PIPELINE TEST")
    print("="*60)
    
    # Step 1: Check if trace file was created
    print("\nSTEP 1: Callsite Tracing")
    print("-" * 40)
    
    if os.path.exists(output_path):
        print(f"✅ Trace file created: {output_path}")
        try:
            with open(output_path) as f:
                data = json.load(f)
            print(f"   Contains {len(data)} callsite entries")
            
            # Show some sample entries
            print("\nSample callsite entries:")
            for i, (key, value) in enumerate(data.items()):
                if i >= 3:  # Show first 3 entries
                    break
                print(f"  {key}: count={value['count']}, ops={value.get('ops', [])[:3]}")
        except json.JSONDecodeError:
            print("❌ Trace file is not valid JSON!")
            return
    else:
        print("❌ Trace file not created!")
        return
    
    # Step 2: Compress the trace
    print("\nSTEP 2: Trace Compression")
    print("-" * 40)
    
    compressed_file = "callsites_compressed.json"
    cmd = [
        sys.executable, "-m", "opguard.callsite_tracer.compress_callsite",
        "-i", output_path,
        "-o", compressed_file,
        "--threshold-num", "1"
    ]
    
    try:
        import subprocess
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ Trace compression completed successfully")
        print(f"   Compressed file: {compressed_file}")
        
        # Check compressed file
        if os.path.exists(compressed_file):
            with open(compressed_file) as f:
                compressed_data = json.load(f)
            print(f"   Contains {len(compressed_data)} compressed entries")
            
            # Show sample compressed entries
            print("\nSample compressed entries:")
            for i, (key, value) in enumerate(compressed_data.items()):
                if i >= 3:  # Show first 3 entries
                    break
                print(f"  {key}: count={value['count']}")
        else:
            print("❌ Compressed file not created!")
            return
    except subprocess.CalledProcessError as e:
        print(f"❌ Trace compression failed: {e}")
        return
    
    # Step 3: Generate instrumentation config
    print("\nSTEP 3: Instrumentation Config Generation")
    print("-" * 40)
    
    config_file = "instrumentation_config.yaml"
    cmd = [
        sys.executable, "-m", "opguard.callsite_to_instrumentation",
        "--callsite-file", compressed_file,
        "--output", config_file
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ Config generation completed successfully")
        print(f"   Config file: {config_file}")
        
        # Check config file
        if os.path.exists(config_file):
            with open(config_file) as f:
                config_content = f.read()
            print(f"   Config size: {len(config_content)} characters")
            
            # Show config content
            print("\nGenerated config:")
            print(config_content)
        else:
            print("❌ Config file not created!")
            return
    except subprocess.CalledProcessError as e:
        print(f"❌ Config generation failed: {e}")
        return
    
    # Step 4: Run instrumentation
    print("\nSTEP 4: Code Instrumentation")
    print("-" * 40)
    
    cmd = [
        sys.executable, "-m", "opguard.instrument_all_files", config_file
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ Instrumentation completed successfully")
        
        # Check if instrumented files were created
        instrumented_dir = Path("opguard-traces")
        if instrumented_dir.exists():
            instrumented_files = list(instrumented_dir.rglob("*.py"))
            print(f"   Created {len(instrumented_files)} instrumented files")
            
            # Show some instrumented files
            for f in instrumented_files[:5]:  # Show first 5
                print(f"     {f}")
        else:
            print("❌ No instrumented files created!")
            return
    except subprocess.CalledProcessError as e:
        print(f"❌ Instrumentation failed: {e}")
        return
    
    # Step 5: Run instrumented code and validate trace dumping
    print("\nSTEP 5: Instrumented Code Execution and Trace Validation")
    print("-" * 40)
    
    # Find the instrumented version of our test file
    instrumented_file = instrumented_dir / "test_pipeline_with_tracer.py"
    
    if not instrumented_file.exists():
        print(f"❌ Instrumented file not found: {instrumented_file}")
        return
    
    print(f"✅ Found instrumented file: {instrumented_file}")
    
    # Run the instrumented program
    cmd = [sys.executable, str(instrumented_file)]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ Instrumented program executed successfully")
        
        # Check for trace output
        trace_files = list(instrumented_dir.glob("*.json"))
        
        print(f"\nTrace files found: {len(trace_files)}")
        for trace_file in trace_files:
            print(f"  {trace_file}")
            if trace_file.stat().st_size > 0:
                print(f"    Size: {trace_file.stat().st_size} bytes")
                try:
                    with open(trace_file) as f:
                        data = json.load(f)
                    print(f"    Contains {len(data)} entries")
                except json.JSONDecodeError:
                    print("    Invalid JSON format")
            else:
                print("    Empty file")
        
        # Validate that we have some trace output
        if not trace_files:
            print("❌ No trace files generated!")
            return
        
        # Check if any trace files have content
        has_content = False
        for trace_file in trace_files:
            if trace_file.stat().st_size > 0:
                has_content = True
                break
        
        if not has_content:
            print("❌ All trace files are empty!")
            return
        
        print("✅ Trace files generated successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Instrumented program execution failed: {e}")
        return
    
    # Final result
    print("\n" + "="*60)
    print("FINAL RESULT")
    print("="*60)
    print("✅ OpGuard end-to-end pipeline test PASSED!")
    print("All steps completed successfully:")
    print("  1. Callsite tracing")
    print("  2. Trace compression")
    print("  3. Instrumentation config generation")
    print("  4. Code instrumentation")
    print("  5. Instrumented code execution and trace validation")
    print("\n🎉 OpGuard pipeline is working correctly!")
    
    # Cleanup
    print("\nCleaning up test artifacts...")
    cleanup_files = [
        compressed_file,
        config_file,
        "simple_model_checkpoint.pt"
    ]
    
    for file in cleanup_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"  Removed: {file}")
    
    print("✅ Cleanup completed!")

if __name__ == "__main__":
    main() 