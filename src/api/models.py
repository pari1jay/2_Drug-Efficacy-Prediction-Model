"""
API data models for drug efficacy prediction
"""
from pydantic import BaseModel, Field
from typing import Dict

class DrugProperties(BaseModel):
    """Input model for drug properties"""
    molecular_weight: float = Field(..., ge=50, le=1000, description="Molecular weight (50-1000)")
    logp: float = Field(..., ge=-2, le=6, description="LogP lipophilicity (-2 to 6)")
    polar_surface_area: float = Field(..., ge=0, le=200, description="Polar surface area (0-200)")
    num_rotatable_bonds: int = Field(..., ge=0, le=20, description="Number of rotatable bonds (0-20)")
    num_aromatic_rings: int = Field(..., ge=0, le=10, description="Number of aromatic rings (0-10)")
    num_hydrogen_donors: int = Field(..., ge=0, le=20, description="Number of hydrogen donors (0-20)")
    num_hydrogen_acceptors: int = Field(..., ge=0, le=20, description="Number of hydrogen acceptors (0-20)")

    class Config:
        schema_extra = {
            "example": {
                "molecular_weight": 300.0,
                "logp": 2.5,
                "polar_surface_area": 80.0,
                "num_rotatable_bonds": 5,
                "num_aromatic_rings": 2,
                "num_hydrogen_donors": 2,
                "num_hydrogen_acceptors": 4
            }
        }

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    efficacy_score: float = Field(..., description="Predicted efficacy score (0-1)")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    interpretation: str = Field(..., description="Human-readable interpretation")
    input_properties: Dict = Field(..., description="Input drug properties")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    message: str
    model_loaded: bool