import os
import requests
from langchain.chat_models import init_chat_model 
from langchain.agents import create_agent 
from dotenv import load_dotenv
from langchain.tools import tool
load_dotenv(".env")

api_key = os.environ["GEMINI_API_KEY"]
model = init_chat_model(
    "google_genai:gemini-2.5-flash",
    api_key = api_key
)

ticker_api_key = os.environ["TWELVEDATA_API_KEY"]

@tool
def get_stock_price(ticker:str) -> str:
    """Get the current market price for a stock. Example: ticker='AAPL'"""
    url = "https://api.twelvedata.com/price"
    params = {
        "symbol" : ticker,
        "apikey" : ticker_api_key
    }
    response = requests.get(url,params=params)
    data = response.json()
    return f"{ticker} current price: ${float(data['price']):.2f}"

@tool 
def get_price_history(ticker:str)->str:
    """Get the last 10 days of closing prices for a stock. Example: ticker='AAPL'"""
    url = "https://api.twelvedata.com/time_series"
    params={
        "symbol":ticker,
        "interval":"1day",
        "outputsize":"10",
        "apikey": ticker_api_key
    }
    res = requests.get(url,params=params)
    data = res.json()
    if "values" not in data:
        return f"Could not fetch price history for {ticker}: {data.get('message', 'Unknown error')}" 
    prices = [
        f"{item['datetime']} : ${float(item['close']):.2f}"
        for item in reversed(data['values'])
    ]
    return f"{ticker} closing prices in 10 days:\n" + "\n".join(prices)

@tool 
def calculate_rsi(ticker:str)->str:
    """Calculate the current RSI for a stock to check if it is overbought or oversold. Example: ticker='AAPL'"""
    url = "https://api.twelvedata.com/rsi"
    params = {
        "symbol" : ticker,
        "interval":"day",
        "time_period":"14",
        "apikey": ticker_api_key
    }
    res = requests.get(url,params=params)
    data = res.json()
    if "values" not in data:
        return f"Could not fetch RSI for {ticker}: {data.get('message', 'Unknown error')}"
    rsi = float(data["values"][0]["rsi"])
    if rsi > 70:
        signal = "Overbought — price may drop soon"
    elif rsi < 30:
        signal = "Oversold — price may rise soon"
    else:
        signal = "Normal"
    return f"{ticker}  RSI(14): {rsi:.1f} → {signal}" 

@tool 
def get_sma(ticker:str) -> str:
    """Get the 50-day and 200-day Simple Moving Averages to check the long-term trend. Example: ticker='AAPL'"""
    base_params={
        "symbol":ticker,
        "interval":"1day",
        "apikey":ticker_api_key
    }
    res_50 = requests.get(
        "https://api.twelvedata.com/sma",
        params = {**base_params,"time_period":"50"}
    )
    res_200 = requests.get(
        "https://api.twelvedata.com/sma",
        params = {**base_params,"time_period":"200"}
    )
    data_50 = res_50.json()
    data_200 = res_200.json()

    if "values" not in data_50 or "values" not in data_200:
        return f"Could not fetch SMA for {ticker}"
    
    sma_50 = float(data_50["values"][0]["sma"])
    sma_200 = float(data_200["values"][0]["sma"])

    if sma_50 > sma_200:
        signal = "Golden Cross — long-term uptrend (Bullish)"
    else:
        signal = "Death Cross — long-term downtrend (Bearish)"
    return f"{ticker} 50-day SMA: ${sma_50:.2f} | 200-day SMA: ${sma_200:.2f} | {signal}" 

system_prompt = """You are a stock trading analyst.

When asked to analyse a stock:
1. Get the current price using get_stock_price
2. Get the recent price history using get_price_history
3. Calculate RSI using calculate_rsi
4. Get the SMA using get_sma
5. Combine all the data and provide:
   - DECISION: BUY / SELL / HOLD
   - CONFIDENCE: High / Medium / Low
   - A short rationale based on the signals

Do not skip any tool. Always run all five before giving a decision."""

agent = create_agent(
    model=model,
    tools = [get_stock_price,get_price_history,calculate_rsi,get_sma],
    system_prompt = system_prompt
)

user_query = "Should I buy Apple stock?"
user_query1 = "Apple vs Microsoft — which looks technically stronger right now?"

response = agent.invoke({
    "messages":[
        {
            "role":"user",
            "content":user_query
        }
    ]
})
print(response["messages"][-1].content)

response1 = agent.invoke({
    "messages":[
        {
            "role":"user",
            "content":user_query1
        }
    ]
})
print(response1["messages"][-1].content)

# [0]["text"]
