#!/usr/bin/env python3
"""
LangChain Output Parsers Demonstration
Environmental Data Processing Examples

Required packages:
pip install langchain>=0.1.0 langchain-core pydantic openai

Recommended LangChain version: 0.1.0+
"""
import os
import json
import re
from typing import List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator

# LangChain imports
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    CommaSeparatedListOutputParser,
    ListOutputParser,
    XMLOutputParser,
    MarkdownListOutputParser,
    NumberedListOutputParser
)
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os

# Initialize OpenAI LLM
def create_openai_llm():
    """Create and return an OpenAI LLM instance"""
    # You can set your API key in environment variable OPENAI_API_KEY
    # or pass it directly here (not recommended for production)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY environment variable not set.")
        print("Please set it with: export OPENAI_API_KEY='your-api-key-here'")
        print("Using gpt-4o-mini model with temperature=0.3 for consistent outputs")

    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,  # Lower temperature for more consistent outputs
        api_key=api_key
    )

# Pydantic models for structured data validation
class EmissionData(BaseModel):
    """Model for CO2 emission data with validation"""
    co2_levels: float = Field(description="CO2 levels in parts per million")
    temperature_rise: float = Field(description="Temperature rise in Celsius")
    sea_level_rise_mm_per_year: float = Field(description="Sea level rise in mm per year")
    arctic_ice_loss_percent: float = Field(description="Arctic ice loss percentage")
    renewable_energy_percent: float = Field(description="Renewable energy percentage")
    deforestation_rate_hectares_per_year: int = Field(description="Deforestation rate in hectares per year")

    @validator('co2_levels')
    def validate_co2_levels(cls, v):
        if v < 280 or v > 500:
            raise ValueError('CO2 levels should be between 280-500 ppm for realistic values')
        return v

    @validator('temperature_rise')
    def validate_temperature(cls, v):
        if v < 0 or v > 5:
            raise ValueError('Temperature rise should be between 0-5°C')
        return v

class SustainabilityReport(BaseModel):
    """Comprehensive sustainability report model"""
    report_title: str = Field(description="Title of the sustainability report")
    assessment_date: datetime = Field(default_factory=datetime.now, description="Date of assessment")
    carbon_footprint_tons: float = Field(description="Carbon footprint in tons CO2 equivalent")
    renewable_percentage: float = Field(ge=0, le=100, description="Percentage of renewable energy use")
    waste_reduction_percent: float = Field(description="Waste reduction percentage")
    water_conservation_liters: int = Field(description="Water conserved in liters")
    sustainability_score: int = Field(ge=0, le=100, description="Overall sustainability score")
    recommendations: List[str] = Field(description="List of sustainability recommendations")

def demonstrate_str_output_parser():
    """Demonstrate StrOutputParser for basic text processing"""
    print("=" * 60)
    print("1. StrOutputParser - Basic String Output")
    print("=" * 60)

    # Create parser and prompt
    parser = StrOutputParser()
    prompt = PromptTemplate(
        template="Provide a brief summary of current climate change impacts: {topic}",
        input_variables=["topic"]
    )

    # Create chain with real OpenAI LLM
    llm = create_openai_llm()
    chain = prompt | llm | parser

    # Execute
    result = chain.invoke({"topic": "global warming effects"})
    print(f"Input: Global warming effects summary")
    print(f"Output: {result}")
    print(f"Output Type: {type(result)}")
    print()

def demonstrate_json_output_parser():
    """Demonstrate JSONOutputParser for structured environmental data"""
    print("=" * 60)
    print("2. JSONOutputParser - Environmental Metrics")
    print("=" * 60)

    # Create parser with format instructions
    parser = JsonOutputParser()
    prompt = PromptTemplate(
        template="""Provide environmental metrics in JSON format: {query}

        CRITICAL: Return ONLY valid JSON. NO underscores in numbers (use 10000000 not 10_000_000).
        Example: {{"deforestation": 10000000, "co2": 415}}

        {format_instructions}""",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    # Create chain with real OpenAI LLM
    llm = create_openai_llm()
    chain = prompt | llm | parser

    # Execute with error handling
    try:
        result = chain.invoke({"query": "current global environmental metrics"})
        print(f"Input: Current global environmental metrics")
        print(f"Output: {json.dumps(result, indent=2)}")
        print(f"Output Type: {type(result)}")
        print(f"CO2 Levels: {result.get('co2_levels', 'N/A')} ppm")
    except Exception as e:
        print(f"JSON Parser Error: {str(e)[:100]}...")
        print("Attempting fallback with fixed JSON...")

        # Fallback: Get raw response and fix it
        llm_only_chain = prompt | llm
        raw_result = llm_only_chain.invoke({"query": "current global environmental metrics"})

        # Fix numeric underscores and parse
        try:
            fixed_json = re.sub(r'(\d+)_(\d+)', r'\1\2', str(raw_result))
            parsed_result = json.loads(fixed_json)
            print(f"✅ Fixed JSON Output: {json.dumps(parsed_result, indent=2)}")
            print(f"Output Type: {type(parsed_result)}")
        except Exception as parse_error:
            print(f"❌ Could not fix JSON: {parse_error}")
            print(f"Raw output: {raw_result[:200]}...")

    print()

def demonstrate_csv_output_parser():
    """Demonstrate CommaSeparatedListOutputParser for climate data"""
    print("=" * 60)
    print("3. CommaSeparatedListOutputParser - Climate Data List")
    print("=" * 60)

    # Create parser
    parser = CommaSeparatedListOutputParser()
    prompt = PromptTemplate(
        template="""Provide a comma-separated list of climate indicators: {request}

        {format_instructions}""",
        input_variables=["request"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    # Create chain with real OpenAI LLM
    llm = create_openai_llm()
    chain = prompt | llm | parser

    # Execute
    result = chain.invoke({"request": "key climate indicators for 2024"})
    print(f"Input: Key climate indicators for 2024")
    print(f"Output: {result}")
    print(f"Output Type: {type(result)}")
    print(f"Number of items: {len(result)}")
    print()

def demonstrate_xml_output_parser():
    """Demonstrate XMLOutputParser for structured environmental reports"""
    print("=" * 60)
    print("4. XMLOutputParser - Environmental Report")
    print("=" * 60)

    # Create parser
    parser = XMLOutputParser()
    prompt = PromptTemplate(
        template="""Generate an environmental report in XML format: {topic}

        Please format your response as XML with elements for title, summary, findings, and urgency_level.""",
        input_variables=["topic"]
    )

    # Create chain with real OpenAI LLM
    llm = create_openai_llm()
    chain = prompt | llm | parser

    # Execute
    result = chain.invoke({"topic": "global environmental status"})
    print(f"Input: Global environmental status report")
    print(f"Output: {result}")
    print(f"Output Type: {type(result)}")
    if isinstance(result, dict):
        # Handle XML parser output structure
        if 'report' in result:
            report_data = result['report']
            if isinstance(report_data, list) and len(report_data) > 0:
                title_item = next((item for item in report_data if 'title' in item), {})
                print(f"Title: {title_item.get('title', 'N/A')}")
            elif isinstance(report_data, dict):
                print(f"Title: {report_data.get('title', 'N/A')}")
    print()

def demonstrate_simple_json_validation():
    """Demonstrate simple JSON validation for environmental data"""
    print("=" * 60)
    print("5. Simple JSON Validation - Environmental Data")
    print("=" * 60)

    # Create parser
    parser = JsonOutputParser()
    prompt = PromptTemplate(
        template="""Provide environmental data in JSON format: {request}

        Please return valid JSON with keys: temperature, co2_levels, sea_level_rise""",
        input_variables=["request"]
    )

    # Create chain with real OpenAI LLM
    llm = create_openai_llm()
    chain = prompt | llm | parser

    # Execute
    try:
        result = chain.invoke({"request": "current climate metrics"})
        print(f"Input: Current climate metrics")
        print(f"Output: {json.dumps(result, indent=2)}")
        print(f"Output Type: {type(result)}")

        # Basic validation
        required_keys = ['co2_levels', 'temperature_rise']
        missing_keys = [key for key in required_keys if key not in result]
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        else:
            print("JSON validation: PASSED")
    except Exception as e:
        print(f"Parsing Error: {e}")
    print()

def demonstrate_markdown_list_parser():
    """Demonstrate MarkdownListOutputParser for environmental recommendations"""
    print("=" * 60)
    print("6. MarkdownListOutputParser - Environmental Recommendations")
    print("=" * 60)

    # Create parser
    parser = MarkdownListOutputParser()
    prompt = PromptTemplate(
        template="""Provide environmental recommendations as a markdown list: {topic}

        Format your response as a markdown list with - bullets.""",
        input_variables=["topic"]
    )

    # Create chain with real OpenAI LLM
    llm = create_openai_llm()
    chain = prompt | llm | parser

    try:
        result = chain.invoke({"topic": "corporate sustainability initiatives"})
        print(f"Input: Corporate sustainability initiatives")
        print(f"Output: {result}")
        print(f"Output Type: {type(result)}")
        print(f"Number of recommendations: {len(result) if isinstance(result, list) else 'N/A'}")
    except Exception as e:
        print(f"Parsing Error: {e}")
    print()

def main():
    """Main function to run all demonstrations"""
    print("LangChain Output Parsers Environmental Data Demo")
    print("=" * 60)
    print("Recommended LangChain version: 0.1.0+")
    print("Install: pip install langchain>=0.1.0 langchain-core pydantic")
    print()

    # Run all demonstrations
    demonstrate_str_output_parser()
    demonstrate_json_output_parser()
    demonstrate_csv_output_parser()
    demonstrate_xml_output_parser()
    demonstrate_simple_json_validation()
    demonstrate_markdown_list_parser()

    print("=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("• StrOutputParser: Simple text processing")
    print("• JsonOutputParser: Structured data with flexible schema")
    print("• CSVOutputParser: Tabular data for analysis")
    print("• StructuredOutputParser: Predefined response schemas")
    print("• PydanticOutputParser: Type-safe validation and complex models")
    print("\nNext Steps:")
    print("• Replace MockLLM with actual LLM (OpenAI, Anthropic, etc.)")
    print("• Add error handling for production use")
    print("• Customize Pydantic models for your specific use case")

if __name__ == "__main__":
    main()