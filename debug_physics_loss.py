#!/usr/bin/env python3
"""
Debug script to test physics loss computation without full training.
Run this to verify the gradient computation fix works.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from training_utils import PhysicsLoss, SimplePhysicsLoss

def test_physics_loss():
    """Test physics loss computation with sample data."""
    print("=== TESTING PHYSICS LOSS COMPUTATION ===")
    
    # Create sample data
    batch_size = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Sample inputs
    t = torch.randn(batch_size, 1, requires_grad=True, device=device)
    xy_pred = torch.randn(batch_size, 2, device=device)
    params = torch.randn(batch_size, 3, device=device)  # K, B, tau
    
    print(f"Input shapes: t={t.shape}, xy_pred={xy_pred.shape}, params={params.shape}")
    print(f"t.requires_grad: {t.requires_grad}")
    
    # Test SimplePhysicsLoss (should work)
    print("\n--- Testing SimplePhysicsLoss ---")
    try:
        simple_loss_fn = SimplePhysicsLoss(weight=0.1).to(device)
        simple_loss = simple_loss_fn(t, xy_pred, params)
        print(f"✅ SimplePhysicsLoss computed: {simple_loss.item():.6f}")
        
        # Test backward pass
        simple_loss.backward(retain_graph=True)
        print(f"✅ SimplePhysicsLoss backward pass successful")
        
    except Exception as e:
        print(f"❌ SimplePhysicsLoss failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test PhysicsLoss (the problematic one)
    print("\n--- Testing PhysicsLoss ---")
    try:
        physics_loss_fn = PhysicsLoss(weight=0.1).to(device)
        physics_loss = physics_loss_fn(t, xy_pred, params)
        print(f"✅ PhysicsLoss computed: {physics_loss.item():.6f}")
        
        # Test backward pass
        physics_loss.backward()
        print(f"✅ PhysicsLoss backward pass successful")
        
    except Exception as e:
        print(f"❌ PhysicsLoss failed: {e}")
        import traceback
        traceback.print_exc()

def test_model_integration():
    """Test physics loss with actual model output."""
    print("\n=== TESTING MODEL INTEGRATION ===")
    
    try:
        from improved_models import SubjectPINN
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create small model for testing
        model = SubjectPINN(
            subject_ids=['test_subject'],
            hidden_dims=[64, 64],
            param_bounds={'K': (500.0, 3000.0), 'B': (20.0, 150.0), 'tau': (0.05, 0.4)}
        ).to(device)
        
        # Create sample data
        batch_size = 5
        t = torch.randn(batch_size, 1, requires_grad=True, device=device)
        subject_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Forward pass
        xy_pred, params = model(t, subject_idx)
        print(f"Model output shapes: xy_pred={xy_pred.shape}, params={params.shape}")
        print(f"xy_pred requires_grad: {xy_pred.requires_grad}")
        print(f"params requires_grad: {params.requires_grad}")
        
        # Test SimplePhysicsLoss with model
        simple_loss_fn = SimplePhysicsLoss(weight=0.1).to(device)
        simple_loss = simple_loss_fn(t, xy_pred, params)
        print(f"✅ SimplePhysicsLoss with model: {simple_loss.item():.6f}")
        
        # Test PhysicsLoss with model
        physics_loss_fn = PhysicsLoss(weight=0.1).to(device)
        physics_loss = physics_loss_fn(t, xy_pred, params)
        print(f"✅ PhysicsLoss with model: {physics_loss.item():.6f}")
        
        # Test backward pass
        total_loss = simple_loss + physics_loss
        total_loss.backward()
        print(f"✅ Combined backward pass successful")
        
    except Exception as e:
        print(f"❌ Model integration test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_physics_loss()
    test_model_integration()
    print("\n=== DEBUG TEST COMPLETE ===")