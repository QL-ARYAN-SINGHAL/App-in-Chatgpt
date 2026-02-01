# ABOUTME: MCP server for the "Marathon Finder" ChatGPT app.
# ABOUTME: Provides tools to search marathons by distance, location, rating, and price.
# ABOUTME: Integrates Stripe for payment processing.

from __future__ import annotations

import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import mcp.types as types
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

# Configuration
ROOT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = ROOT_DIR.parent / "web"  # web folder is at project root
TEMPLATE_URI = "ui://widget/main.html"
MIME_TYPE = "text/html+skybridge"

# Production deployment configuration
WIDGET_DOMAIN = os.environ.get("WIDGET_DOMAIN", "https://web-sandbox.oaiusercontent.com")
PORT = int(os.environ.get("PORT", 8000))

# Stripe configuration
STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_ENABLED = bool(STRIPE_SECRET_KEY)

# Import Stripe if available
if STRIPE_ENABLED:
    try:
        import stripe
        stripe.api_key = STRIPE_SECRET_KEY
        print("[Marathon Finder] Stripe integration enabled")
    except ImportError:
        print("[Marathon Finder] Stripe library not installed. Run: pip install stripe")
        STRIPE_ENABLED = False
else:
    print("[Marathon Finder] Stripe not configured. Set STRIPE_SECRET_KEY environment variable.")

# In-memory registration storage (in production, use a real database)
REGISTRATIONS = []


# Local marathon database (in production, this would be a real database)
MARATHON_DATABASE = [
    {
        "id": 1,
        "name": "NYC Marathon",
        "city": "New York",
        "state": "NY",
        "distance": "42.2 km",
        "distance_km": 42.2,
        "price": "$255",
        "price_usd": 255,
        "rating": 4.8,
        "date": "Nov 3, 2025",
        "participants": "50,000+",
        "latitude": 40.7128,
        "longitude": -74.0060,
    },
    {
        "id": 2,
        "name": "Boston Marathon",
        "city": "Boston",
        "state": "MA",
        "distance": "42.2 km",
        "distance_km": 42.2,
        "price": "$240",
        "price_usd": 240,
        "rating": 4.9,
        "date": "Apr 21, 2025",
        "participants": "30,000+",
        "latitude": 42.3601,
        "longitude": -71.0589,
    },
    {
        "id": 3,
        "name": "Chicago Marathon",
        "city": "Chicago",
        "state": "IL",
        "distance": "42.2 km",
        "distance_km": 42.2,
        "price": "$230",
        "price_usd": 230,
        "rating": 4.7,
        "date": "Oct 12, 2025",
        "participants": "45,000+",
        "latitude": 41.8781,
        "longitude": -87.6298,
    },
    {
        "id": 4,
        "name": "Los Angeles Marathon",
        "city": "Los Angeles",
        "state": "CA",
        "distance": "42.2 km",
        "distance_km": 42.2,
        "price": "$210",
        "price_usd": 210,
        "rating": 4.6,
        "date": "Mar 23, 2025",
        "participants": "25,000+",
        "latitude": 34.0522,
        "longitude": -118.2437,
    },
    {
        "id": 5,
        "name": "San Francisco Half Marathon",
        "city": "San Francisco",
        "state": "CA",
        "distance": "21.1 km",
        "distance_km": 21.1,
        "price": "$150",
        "price_usd": 150,
        "rating": 4.5,
        "date": "Jul 27, 2025",
        "participants": "20,000+",
        "latitude": 37.7749,
        "longitude": -122.4194,
    },
    {
        "id": 6,
        "name": "Miami Marathon",
        "city": "Miami",
        "state": "FL",
        "distance": "42.2 km",
        "distance_km": 42.2,
        "price": "$185",
        "price_usd": 185,
        "rating": 4.4,
        "date": "Feb 9, 2025",
        "participants": "18,000+",
        "latitude": 25.7617,
        "longitude": -80.1918,
    },
    {
        "id": 7,
        "name": "Seattle Rock 'n' Roll Marathon",
        "city": "Seattle",
        "state": "WA",
        "distance": "42.2 km",
        "distance_km": 42.2,
        "price": "$195",
        "price_usd": 195,
        "rating": 4.6,
        "date": "Jun 22, 2025",
        "participants": "15,000+",
        "latitude": 47.6062,
        "longitude": -122.3321,
    },
    {
        "id": 8,
        "name": "Austin Marathon",
        "city": "Austin",
        "state": "TX",
        "distance": "42.2 km",
        "distance_km": 42.2,
        "price": "$175",
        "price_usd": 175,
        "rating": 4.5,
        "date": "Feb 16, 2025",
        "participants": "16,000+",
        "latitude": 30.2672,
        "longitude": -97.7431,
    },
    {
        "id": 9,
        "name": "Denver Half Marathon",
        "city": "Denver",
        "state": "CO",
        "distance": "21.1 km",
        "distance_km": 21.1,
        "price": "$120",
        "price_usd": 120,
        "rating": 4.3,
        "date": "Oct 19, 2025",
        "participants": "12,000+",
        "latitude": 39.7392,
        "longitude": -104.9903,
    },
    {
        "id": 10,
        "name": "Philadelphia Marathon",
        "city": "Philadelphia",
        "state": "PA",
        "distance": "42.2 km",
        "distance_km": 42.2,
        "price": "$165",
        "price_usd": 165,
        "rating": 4.4,
        "date": "Nov 23, 2025",
        "participants": "30,000+",
        "latitude": 39.9526,
        "longitude": -75.1652,
    },
    {
        "id": 11,
        "name": "Portland 10K",
        "city": "Portland",
        "state": "OR",
        "distance": "10 km",
        "distance_km": 10,
        "price": "$75",
        "price_usd": 75,
        "rating": 4.2,
        "date": "May 18, 2025",
        "participants": "8,000+",
        "latitude": 45.5152,
        "longitude": -122.6784,
    },
    {
        "id": 12,
        "name": "Phoenix Marathon",
        "city": "Phoenix",
        "state": "AZ",
        "distance": "42.2 km",
        "distance_km": 42.2,
        "price": "$155",
        "price_usd": 155,
        "rating": 4.3,
        "date": "Feb 22, 2025",
        "participants": "10,000+",
        "latitude": 33.4484,
        "longitude": -112.0740,
    },
]


@lru_cache(maxsize=None)
def load_widget_html() -> str:
    """Load and cache the widget HTML."""
    html_path = ASSETS_DIR / "widget.html"
    return html_path.read_text(encoding="utf8")


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points using Haversine formula (in km)."""
    from math import radians, sin, cos, sqrt, atan2

    R = 6371  # Earth's radius in kilometers

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c

    return distance


def create_stripe_checkout_session(
    marathon_id: int,
    marathon_name: str,
    price_usd: float,
    customer_email: str,
    customer_name: str,
    metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """Create a Stripe Checkout Session and return the payment URL."""
    if not STRIPE_ENABLED:
        return {
            "success": False,
            "error": "Stripe is not configured. Please set STRIPE_SECRET_KEY environment variable."
        }
    
    try:
        # Create Stripe Checkout Session
        session = stripe.checkout.Session.create(
    payment_method_types=["card"],
    line_items=[{
        "price_data": {
            "currency": "usd",
            "product_data": {
                "name": f"{marathon_name} - Registration",
            },
            "unit_amount": int(price_usd * 100),
        },
        "quantity": 1,
    }],
    mode="payment",
    customer_email=customer_email,
    metadata=metadata,

    success_url="https://example.com/success",
    cancel_url="https://example.com/cancel",
)

        return {
            "success": True,
            "payment_url": session.url,
            "session_id": session.id
        }
    
    except Exception as e:
        print(f"[Marathon Finder] Stripe error: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


def search_marathons(
    city: Optional[str] = None,
    state: Optional[str] = None,
    min_rating: Optional[float] = None,
    max_rating: Optional[float] = None,
    distance_km: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    max_distance_from_location: Optional[float] = None,
    user_latitude: Optional[float] = None,
    user_longitude: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Search marathons based on various filters."""
    results = MARATHON_DATABASE.copy()

    # Filter by city
    if city:
        results = [m for m in results if city.lower() in m["city"].lower()]

    # Filter by state
    if state:
        results = [m for m in results if state.upper() == m["state"].upper()]

    # Filter by rating
    if min_rating is not None:
        results = [m for m in results if m["rating"] >= min_rating]
    if max_rating is not None:
        results = [m for m in results if m["rating"] <= max_rating]

    # Filter by distance (race distance, not location distance)
    if distance_km:
        # Parse distance string like "42.2 km", "21.1 km", "10 km"
        distance_value = float(distance_km.replace("km", "").strip())
        results = [m for m in results if m["distance_km"] == distance_value]

    # Filter by price
    if min_price is not None:
        results = [m for m in results if m["price_usd"] >= min_price]
    if max_price is not None:
        results = [m for m in results if m["price_usd"] <= max_price]

    # Filter by proximity to user location
    if max_distance_from_location and user_latitude and user_longitude:
        filtered_results = []
        for marathon in results:
            distance = calculate_distance(
                user_latitude, user_longitude, marathon["latitude"], marathon["longitude"]
            )
            if distance <= max_distance_from_location:
                marathon_copy = marathon.copy()
                marathon_copy["distance_from_user"] = round(distance, 1)
                filtered_results.append(marathon_copy)
        results = filtered_results

    # Sort by rating (highest first)
    results.sort(key=lambda x: x["rating"], reverse=True)

    return results


def tool_meta() -> Dict[str, Any]:
    """Return standard tool metadata with CSP for production deployment."""
    return {
        "openai/outputTemplate": TEMPLATE_URI,
        "openai/widgetAccessible": True,
        "openai/widgetCSP": {
            "connect_domains": [],  # Empty - widget doesn't make API calls
            "resource_domains": [],  # Empty - all assets are inline
        },
        "openai/widgetDomain": WIDGET_DOMAIN,
    }


# Initialize FastMCP with stateless HTTP mode
mcp = FastMCP(
    name="marathon-finder",
    stateless_http=True,
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


# Register widget as MCP resource
@mcp._mcp_server.list_resources()
async def _list_resources() -> List[types.Resource]:
    return [
        types.Resource(
            name="Marathon Finder Widget",
            uri=TEMPLATE_URI,
            description="Visual interface for marathon search results",
            mimeType=MIME_TYPE,
            _meta=tool_meta(),
        )
    ]


@mcp._mcp_server.list_resource_templates()
async def _list_resource_templates() -> List[types.ResourceTemplate]:
    return [
        types.ResourceTemplate(
            name="Marathon Finder Widget",
            uriTemplate=TEMPLATE_URI,
            description="Visual interface for marathon search results",
            mimeType=MIME_TYPE,
            _meta=tool_meta(),
        )
    ]


async def _handle_read_resource(req: types.ReadResourceRequest) -> types.ServerResult:
    """Handle resource read requests."""
    if str(req.params.uri) != TEMPLATE_URI:
        return types.ServerResult(
            types.ReadResourceResult(
                contents=[],
                _meta={"error": f"Unknown resource: {req.params.uri}"},
            )
        )

    return types.ServerResult(
        types.ReadResourceResult(
            contents=[
                types.TextResourceContents(
                    uri=TEMPLATE_URI,
                    mimeType=MIME_TYPE,
                    text=load_widget_html(),
                    _meta=tool_meta(),
                )
            ]
        )
    )


# Register tool
@mcp._mcp_server.list_tools()
async def _list_tools() -> List[types.Tool]:
    return [
        types.Tool(
            name="search_marathons",
            title="Search Marathons",
            description="Use this tool when user asks to find, search, or look for marathons or races. Search marathons by location (city, state), distance (10km, 21.1km, 42.2km), rating (1-5), and price range. Can also find marathons within a certain distance from user's location. Always use this tool for marathon-related queries.",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name (e.g., 'New York', 'Boston')",
                    },
                    "state": {
                        "type": "string",
                        "description": "State abbreviation (e.g., 'NY', 'CA', 'TX')",
                    },
                    "min_rating": {
                        "type": "number",
                        "description": "Minimum rating (1-5)",
                        "minimum": 1,
                        "maximum": 5,
                    },
                    "max_rating": {
                        "type": "number",
                        "description": "Maximum rating (1-5)",
                        "minimum": 1,
                        "maximum": 5,
                    },
                    "distance_km": {
                        "type": "string",
                        "description": "Race distance in km (e.g., '10', '21.1', '42.2')",
                        "enum": ["10", "21.1", "42.2"],
                    },
                    "min_price": {
                        "type": "number",
                        "description": "Minimum price in USD",
                    },
                    "max_price": {
                        "type": "number",
                        "description": "Maximum price in USD",
                    },
                    "max_distance_from_location": {
                        "type": "number",
                        "description": "Maximum distance from user location in km",
                    },
                    "user_latitude": {
                        "type": "number",
                        "description": "User's latitude for proximity search",
                    },
                    "user_longitude": {
                        "type": "number",
                        "description": "User's longitude for proximity search",
                    },
                },
            },
            _meta=tool_meta(),
            annotations={
                "destructiveHint": False,
                "openWorldHint": False,
                "readOnlyHint": True,
            },
        ),
        types.Tool(
            name="register_for_marathon",
            title="Register for Marathon",
            description="Register a user for a marathon event and generate a Stripe payment link. This tool is called from the widget when a user fills out the registration form.",
            inputSchema={
                "type": "object",
                "properties": {
                    "marathon_id": {
                        "type": "integer",
                        "description": "Marathon ID",
                    },
                    "marathon_name": {
                        "type": "string",
                        "description": "Marathon name",
                    },
                    "marathon_city": {
                        "type": "string",
                        "description": "Marathon city",
                    },
                    "marathon_state": {
                        "type": "string",
                        "description": "Marathon state",
                    },
                    "marathon_date": {
                        "type": "string",
                        "description": "Marathon date",
                    },
                    "price_usd": {
                        "type": "number",
                        "description": "Marathon registration price in USD",
                    },
                    "full_name": {
                        "type": "string",
                        "description": "Registrant's full name",
                    },
                    "email": {
                        "type": "string",
                        "description": "Registrant's email",
                    },
                    "phone": {
                        "type": "string",
                        "description": "Registrant's phone number",
                    },
                    "timestamp": {
                        "type": "string",
                        "description": "Registration timestamp",
                    },
                },
                "required": ["marathon_id", "marathon_name", "price_usd", "full_name", "email", "phone"],
            },
            annotations={
                "destructiveHint": False,
                "openWorldHint": False,
                "readOnlyHint": False,
            },
        ),
    ]


async def _handle_call_tool(req: types.CallToolRequest) -> types.ServerResult:
    """Handle tool invocations."""
    tool_name = req.params.name

    if tool_name == "search_marathons":
        args = req.params.arguments or {}

        # Extract search parameters
        city = args.get("city")
        state = args.get("state")
        min_rating = args.get("min_rating")
        max_rating = args.get("max_rating")
        distance_km = args.get("distance_km")
        min_price = args.get("min_price")
        max_price = args.get("max_price")
        max_distance_from_location = args.get("max_distance_from_location")
        user_latitude = args.get("user_latitude")
        user_longitude = args.get("user_longitude")

        # Search marathons
        marathons = search_marathons(
            city=city,
            state=state,
            min_rating=min_rating,
            max_rating=max_rating,
            distance_km=distance_km,
            min_price=min_price,
            max_price=max_price,
            max_distance_from_location=max_distance_from_location,
            user_latitude=user_latitude,
            user_longitude=user_longitude,
        )

        # Build location string for display
        location_parts = []
        if city:
            location_parts.append(city)
        if state:
            location_parts.append(state)
        location_str = ", ".join(location_parts) if location_parts else "All Locations"

        # Prepare marathon data for widget (clean format)
        marathon_list = []
        for m in marathons[:10]:  # Limit to top 10 results
            marathon_list.append({
                "id": m["id"],
                "name": m["name"],
                "city": m["city"],
                "state": m["state"],
                "distance": m["distance"],
                "price": m["price"],
                "price_usd": m["price_usd"],  # Include numeric price for Stripe
                "rating": m["rating"],
                "date": m.get("date", ""),
                "participants": m.get("participants", ""),
            })

        # Build response text
        if marathons:
            text = f"Found {len(marathons)} marathon(s) matching your criteria"
            if location_parts:
                text += f" in {location_str}"
            text += ". Here are the top results:"
        else:
            text = "No marathons found matching your search criteria. Try adjusting your filters."

        # Widget data - this becomes window.openai.toolOutput in the widget
        widget_data = {
            "marathons": marathon_list,
            "location": location_str,
            "total_results": len(marathons),
        }

        return types.ServerResult(
            types.CallToolResult(
                content=[types.TextContent(type="text", text=text)],
                structuredContent=widget_data,
                _meta=tool_meta(),
            )
        )

    elif tool_name == "register_for_marathon":
        args = req.params.arguments or {}
        
        # Extract registration data
        marathon_id = args.get("marathon_id")
        marathon_name = args.get("marathon_name", "Unknown Marathon")
        price_usd = args.get("price_usd", 0)
        full_name = args.get("full_name")
        email = args.get("email")
        phone = args.get("phone")
        
        # Create metadata for Stripe
        metadata = {
            "marathon_id": str(marathon_id),
            "marathon_name": marathon_name,
            "marathon_city": args.get("marathon_city", ""),
            "marathon_state": args.get("marathon_state", ""),
            "marathon_date": args.get("marathon_date", ""),
            "customer_name": full_name,
            "customer_phone": phone,
            "registration_timestamp": args.get("timestamp", datetime.now().isoformat()),
        }
        
        # Create Stripe checkout session
        stripe_result = create_stripe_checkout_session(
            marathon_id=marathon_id,
            marathon_name=marathon_name,
            price_usd=price_usd,
            customer_email=email,
            customer_name=full_name,
            metadata=metadata
        )
        
        if stripe_result["success"]:
            # Create registration record (pending payment)
            registration = {
                "id": len(REGISTRATIONS) + 1,
                "marathon_id": marathon_id,
                "marathon_name": marathon_name,
                "marathon_city": args.get("marathon_city", ""),
                "marathon_state": args.get("marathon_state", ""),
                "marathon_date": args.get("marathon_date", ""),
                "full_name": full_name,
                "email": email,
                "phone": phone,
                "price_usd": price_usd,
                "timestamp": args.get("timestamp"),
                "status": "pending_payment",
                "stripe_session_id": stripe_result["session_id"],
                "payment_url": stripe_result["payment_url"],
            }
            
            # Store registration
            REGISTRATIONS.append(registration)
            
            # Log for debugging
            print(f"[Marathon Finder] New registration created: {registration}")
            print(f"[Marathon Finder] Payment URL: {stripe_result['payment_url']}")
            print(f"[Marathon Finder] Total registrations: {len(REGISTRATIONS)}")
            
            # Build response with payment link
            text = f"✅ Registration details saved for {full_name}!\n\n"
            text += f"💳 Please complete your payment to finalize your registration for {marathon_name}.\n\n"
            text += f"Click the link below to proceed to secure payment:\n"
            text += f"🔗 {stripe_result['payment_url']}\n\n"
            text += f"After successful payment, you'll receive a confirmation email at {email}."
            
            return types.ServerResult(
                types.CallToolResult(
                    content=[types.TextContent(type="text", text=text)],
                    structuredContent={
                        "success": True,
                        "registration_id": registration["id"],
                        "payment_url": stripe_result["payment_url"],
                        "session_id": stripe_result["session_id"]
                    },
                )
            )
        else:
            # Payment link creation failed
            error_text = f"❌ Unable to generate payment link. Error: {stripe_result.get('error', 'Unknown error')}"
            
            return types.ServerResult(
    types.CallToolResult(
        content=[
            types.TextContent(
                type="text",
                text=error_text
            )
        ],
        isError=True
    )
)



    return types.ServerResult(
        types.CallToolResult(
            content=[types.TextContent(type="text", text=f"Unknown tool: {tool_name}")],
            isError=True,
        )
    )


# Register handlers
mcp._mcp_server.request_handlers[types.CallToolRequest] = _handle_call_tool
mcp._mcp_server.request_handlers[types.ReadResourceRequest] = _handle_read_resource


# Create ASGI app with CORS
app = mcp.streamable_http_app()

try:
    from starlette.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
except ImportError:
    pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)