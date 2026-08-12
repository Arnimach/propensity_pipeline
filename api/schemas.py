from pydantic import BaseModel, Field


# Request Schema
class CustomerFeaturesInput(BaseModel):
    Administrative: int = Field(..., description="Number of administrative sites visited during the session")
    Administrative_Duration: float = Field(..., description="Total time (in seconds) spent on administrative pages")
    Informational: int = Field(..., description="Number of informational pages visited")
    Informational_Duration: float = Field(..., description="Total time (in seconds) spent on informational pages")
    ProductRelated: int = Field(..., description=" Number of product-related pages visited")
    ProductRelated_Duration: float = Field(..., description="Total time (in seconds) spent on product pages")
    BounceRates: float = Field(..., description="Average historical bounce rate of all the \
    unique pages the user visited during this session")
    ExitRates: float = Field(..., description="the average historical exit rate of all the \
    unique pages the user visited during this session")
    PageValues: float = Field(..., description="average financial value of all the unique\
     pages the user visited during this session")
    SpecialDay: float = Field(..., description="how close the session date is to a major shopping event")
    Month: str = Field(..., description="The month of the session")
    Weekend: bool = Field(..., description="Day of the session is weekend")
    OperatingSystems: int = Field(..., description="Categorical representation of the OS they used")
    Browser: int = Field(..., description="Categorical representation of the browser they used")
    Region: int = Field(..., description="Anonymized geographic location identifier of the user.")
    TrafficType: int = Field(..., description="How the user found the website")
    VisitorType: str = Field(..., description="Whether they are a New_Visitor, Returning_Visitor, or Other")

    model_config = {
        "json_schema_extra": {
            "example": {
                "Administrative": 0,
                "Administrative_Duration": 0.0,
                "Informational": 2,
                "Informational_Duration": 45.0,
                "ProductRelated": 18,
                "ProductRelated_Duration": 650.5,
                "BounceRates": 0.01,
                "ExitRates": 0.03,
                "PageValues": 24.5,
                "SpecialDay": 0.0,
                "Month": "Nov",
                "OperatingSystems": 2,
                "Browser": 2,
                "Region": 1,
                "TrafficType": 2,
                "VisitorType": "Returning_Visitor",
                "Weekend": False
            }
        }
    }


# 2. Response Schema: What your API returns
class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Binary classification output (0 or 1)")
    high_value_flag: bool = Field(..., description="True if prediction is 1")
    model_version_used: str = Field(..., description="MLflow registered model version executed")
