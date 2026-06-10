// ==========================================================================
// THE LAST BITE — FRONTEND APPLICATION LOGIC (REACT + BABEL)
// Complete redesign: smooth controls, visual star bar, restaurant badges,
// enhanced welcome state, premium recommendation cards
// ==========================================================================

const { useState, useEffect, useRef } = React;

// Coordinates for mapping locations to map centers
const AREA_COORDINATES = {
    "Delhi": [28.6139, 77.2090],
    "New Delhi": [28.6139, 77.2090],
    "Bangalore": [12.9716, 77.5946],
    "Koramangala": [12.9352, 77.6244],
    "Koramangala 5th Block": [12.9352, 77.6244],
    "Koramangala 8th Block": [12.9385, 77.6137],
    "Koramangala 1st Block": [12.9272, 77.6332],
    "Koramangala 6th Block": [12.9385, 77.6228],
    "Koramangala 7th Block": [12.9302, 77.6190],
    "Koramangala 4th Block": [12.9328, 77.6294],
    "Koramangala 2nd Block": [12.9242, 77.6224],
    "Koramangala 3rd Block": [12.9281, 77.6253],
    "Btm": [12.9166, 77.6101],
    "Btm Layout": [12.9166, 77.6101],
    "Indiranagar": [12.9719, 77.6412],
    "Hsr": [12.9101, 77.6450],
    "Jayanagar": [12.9299, 77.5824],
    "Byresandra, Jayanagar": [12.9299, 77.5824],
    "Jp Nagar": [12.9063, 77.5857],
    "Whitefield": [12.9698, 77.7499],
    "Bannerghatta Road": [12.8943, 77.5986],
    "Banashankari": [12.9250, 77.5360],
    "Kalyan Nagar": [13.0273, 77.6411],
    "Bellandur": [12.9304, 77.6784],
    "Sarjapur Road": [12.9200, 77.6830],
    "Malleshwaram": [13.0031, 77.5643],
    "Richmond Road": [12.9680, 77.5980],
    "Electronic City": [12.8452, 77.6602],
    "Marathahalli": [12.9561, 77.7011],
    "Basavanagudi": [12.9417, 77.5755],
    "Shanti Nagar": [12.9580, 77.5980],
    "Bommanahalli": [12.9030, 77.6240],
    "Rajajinagar": [12.9915, 77.5467],
    "Old Airport Road": [12.9592, 77.6534]
};

const VALID_PREFERENCES = [
    "family-friendly",
    "quick service",
    "romantic",
    "outdoor seating",
    "live music",
    "pet-friendly",
    "vegan options",
    "vegetarian",
    "late night",
    "breakfast",
    "buffet",
    "fine dining",
    "casual dining",
    "street food"
];

// Helper to find closest location to GPS coordinates
const getClosestLocation = (lat, lng, locationsList) => {
    const delhiLat = 28.6139, delhiLng = 77.2090;
    const blrLat = 12.9716, blrLng = 77.5946;
    
    const distToDelhi = Math.sqrt(Math.pow(lat - delhiLat, 2) + Math.pow(lng - delhiLng, 2));
    const distToBlr = Math.sqrt(Math.pow(lat - blrLat, 2) + Math.pow(lng - blrLng, 2));
    
    if (distToDelhi < distToBlr) {
        const match = locationsList.find(loc => loc.toLowerCase().includes("delhi"));
        return match || "Delhi";
    }
    
    let closestArea = "Koramangala";
    let minDistance = Infinity;
    
    for (const [area, coords] of Object.entries(AREA_COORDINATES)) {
        if (area === "Delhi" || area === "New Delhi" || area === "Bangalore") continue;
        const dist = Math.sqrt(Math.pow(lat - coords[0], 2) + Math.pow(lng - coords[1], 2));
        if (dist < minDistance) {
            minDistance = dist;
            closestArea = area;
        }
    }
    
    const match = locationsList.find(loc => loc.toLowerCase() === closestArea.toLowerCase());
    return match || locationsList[0];
};

// ==========================================================================
// REACT SVG ICON SYSTEM
// ==========================================================================
const Icon = ({ name, style, className }) => {
    const paths = {
        sparkles: (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/>
                <path d="m5 3 1 2.5L8.5 6 6 7 5 9.5 4 7 1.5 6 4 5.5z"/>
                <path d="m19 17 1 2.5 2.5.5-2.5 1-1 2.5-1-2.5-2.5-1 2.5-1z"/>
            </svg>
        ),
        sliders: (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="4" x2="4" y1="21" y2="14"/><line x1="4" x2="4" y1="10" y2="3"/><line x1="12" x2="12" y1="21" y2="12"/><line x1="12" x2="12" y1="8" y2="3"/><line x1="20" x2="20" y1="21" y2="16"/><line x1="20" x2="20" y1="12" y2="3"/><line x1="2" x2="6" y1="14" y2="14"/><line x1="10" x2="14" y1="8" y2="8"/><line x1="18" x2="22" y1="16" y2="16"/>
            </svg>
        ),
        compass: (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/>
            </svg>
        ),
        search: (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>
            </svg>
        ),
        map: (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M14.106 5.553a2 2 0 0 0-1.788 0l-3.659 1.83a2 2 0 0 1-1.788 0L3 5v12l3.87 1.935a2 2 0 0 0 1.788 0l3.659-1.83a2 2 0 0 1 1.788 0L21 19V7l-6.894-3.447Z"/>
                <line x1="9" x2="9" y1="7.5" y2="19.5"/><line x1="15" x2="15" y1="4.5" y2="16.5"/>
            </svg>
        ),
        brain: (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"/>
                <path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z"/>
                <path d="M12 5v14"/>
            </svg>
        ),
        activity: (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
            </svg>
        ),
        cpu: (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <rect width="16" height="16" x="4" y="4" rx="2"/>
                <rect width="6" height="6" x="9" y="9" rx="1"/>
                <path d="M9 1v3"/><path d="M15 1v3"/><path d="M9 20v3"/><path d="M15 20v3"/><path d="M20 9h3"/><path d="M20 15h3"/><path d="M1 9h3"/><path d="M1 15h3"/>
            </svg>
        ),
        zap: (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
            </svg>
        ),
        calendar: (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M8 2v4"/><path d="M16 2v4"/><rect width="18" height="18" x="3" y="4" rx="2"/><path d="M3 10h18"/>
            </svg>
        ),
        navigation: (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polygon points="3 11 22 2 13 21 11 13 3 11"/>
            </svg>
        ),
        chefHat: (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M6 18V6a4 4 0 0 1 8 0v12"/><path d="M18 18V9a4 4 0 0 0-8 0v9"/><path d="M3 18h18a1 1 0 0 1 1 1v2a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1v-2a1 1 0 0 1 1-1Z"/>
            </svg>
        ),
        check: (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="20 6 9 17 4 12"/>
            </svg>
        ),
        alertTriangle: (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" x2="12" y1="9" y2="13"/><line x1="12" x2="12.01" y1="17" y2="17"/>
            </svg>
        ),
        utensils: (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M3 2v7c0 1.1.9 2 2 2h4a2 2 0 0 0 2-2V2"/><path d="M7 2v20"/><path d="M21 15V2a5 5 0 0 0-5 5v6c0 1.1.9 2 2 2h3Zm0 0v7"/>
            </svg>
        ),
        globe: (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10"/><path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20"/><path d="M2 12h20"/>
            </svg>
        ),
        wifi: (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 20h.01"/><path d="M2 8.82a15 15 0 0 1 20 0"/><path d="M5 12.86a10 10 0 0 1 14 0"/><path d="M8.5 16.43a5 5 0 0 1 7 0"/>
            </svg>
        ),
        store: (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="m2 7 4.41-4.41A2 2 0 0 1 7.83 2h8.34a2 2 0 0 1 1.42.59L22 7"/><path d="M4 12v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8"/><path d="M15 22v-4a2 2 0 0 0-2-2h-2a2 2 0 0 0-2 2v4"/><path d="M2 7h20"/><path d="M22 7v3a2 2 0 0 1-2 2a2.7 2.7 0 0 1-1.59-.63.7.7 0 0 0-.82 0A2.7 2.7 0 0 1 16 12a2.7 2.7 0 0 1-1.59-.63.7.7 0 0 0-.82 0A2.7 2.7 0 0 1 12 12a2.7 2.7 0 0 1-1.59-.63.7.7 0 0 0-.82 0A2.7 2.7 0 0 1 8 12a2.7 2.7 0 0 1-1.59-.63.7.7 0 0 0-.82 0A2.7 2.7 0 0 1 4 12a2 2 0 0 1-2-2V7"/>
            </svg>
        ),
    };

    const iconSvg = paths[name];
    if (!iconSvg) return null;

    const defaultStyle = {
        width: '20px',
        height: '20px',
        stroke: 'currentColor',
        display: 'inline-block',
        verticalAlign: 'middle',
        ...style
    };

    return React.cloneElement(iconSvg, { style: defaultStyle, className });
};

// ==========================================================================
// MAP COMPONENT (LEAFLET WRAPPER)
// ==========================================================================
const MapView = ({ location, recommendations }) => {
    const mapContainerRef = useRef(null);
    const leafletMapRef = useRef(null);
    const markersGroupRef = useRef(null);

    useEffect(() => {
        if (!mapContainerRef.current) return;

        // Determine center
        let center = AREA_COORDINATES[location];
        if (!center) {
            const matchedKey = Object.keys(AREA_COORDINATES).find(
                key => key.toLowerCase().includes(location.toLowerCase()) || 
                       location.toLowerCase().includes(key.toLowerCase())
            );
            center = matchedKey ? AREA_COORDINATES[matchedKey] : [12.9716, 77.5946];
        }

        // Initialize leaflet map if not already done
        if (!leafletMapRef.current) {
            leafletMapRef.current = L.map(mapContainerRef.current, {
                zoomControl: true,
                scrollWheelZoom: false
            }).setView(center, 14);

            L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
                attribution: '&copy; OpenStreetMap &copy; CARTO'
            }).addTo(leafletMapRef.current);

            markersGroupRef.current = L.featureGroup().addTo(leafletMapRef.current);
        } else {
            leafletMapRef.current.setView(center, 14);
            markersGroupRef.current.clearLayers();
        }

        // Add pins for recommended restaurants with deterministic offset so they spread
        const bounds = [];
        recommendations.forEach((rec, idx) => {
            const angle = (idx * 2 * Math.PI) / Math.max(recommendations.length, 1);
            const radius = 0.003; // Spread radius
            const lat = center[0] + radius * Math.sin(angle);
            const lng = center[1] + radius * Math.cos(angle);
            
            const customIcon = L.divIcon({
                className: 'custom-map-pin',
                html: `<div style="
                    background-color: #cb202d;
                    color: white;
                    width: 32px;
                    height: 32px;
                    border-radius: 50% 50% 50% 0;
                    transform: rotate(-45deg);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-weight: 800;
                    font-size: 0.9rem;
                    border: 2px solid white;
                    box-shadow: 0 0 10px rgba(203, 32, 45, 0.6);
                    margin-left: -16px;
                    margin-top: -32px;
                "><span style="transform: rotate(45deg); display: inline-block;">${rec.rank}</span></div>`,
                iconSize: [32, 32],
                iconAnchor: [16, 32]
            });

            const marker = L.marker([lat, lng], { icon: customIcon }).addTo(markersGroupRef.current);
            marker.bindPopup(`
                <div class="map-popup-title">${rec.restaurant_name}</div>
                <div style="margin: 4px 0; display: flex; align-items: center; gap: 6px;">
                    <span class="map-popup-rating" style="background:#10b981; color:#fff; font-size:0.75rem; padding: 2px 5px; border-radius:4px; font-weight:700;">⭐ ${rec.rating.toFixed(1)}</span>
                    <span class="map-popup-cuisine" style="font-size:0.8rem; color:#6b7194;">${rec.cuisine}</span>
                </div>
                <div style="font-size: 0.75rem; color: #6b7194; margin-top: 4px;">
                    Cost: ${rec.estimated_cost}
                </div>
            `);
            bounds.push([lat, lng]);
        });

        // Zoom map to fit all markers
        if (bounds.length > 0) {
            leafletMapRef.current.fitBounds(bounds, { padding: [40, 40] });
        }
    }, [location, recommendations]);

    return (
        <div className="map-view-card">
            <h4 style={{ marginBottom: '0.8rem', display: 'flex', alignItems: 'center', gap: '0.5rem', fontFamily: 'var(--font-heading)', color: 'var(--text-primary)', fontSize: '1rem', fontWeight: 700 }}>
                <Icon name="map" style={{ color: 'var(--color-primary)', width: '18px', height: '18px' }} />
                Restaurant Map View
            </h4>
            <div ref={mapContainerRef} className="map-container"></div>
        </div>
    );
};

// ==========================================================================
// METRIC SLIDER — RED HORIZONTAL POINTER INDICATOR
// ==========================================================================
const MetricSlider = ({ label, value, maxVal, minVal = 0, formatVal }) => {
    const range = maxVal - minVal;
    const adjustedVal = Math.max(minVal, Math.min(maxVal, value));
    const percentage = ((adjustedVal - minVal) / range) * 100;

    return (
        <div className="metric-section">
            <div className="metric-header">
                <span>{label}</span>
                <span className="val">{formatVal(value)}</span>
            </div>
            <div className="metric-slider-track">
                <div className="metric-slider-fill" style={{ width: `${percentage}%` }}></div>
                <div className="metric-slider-pointer" style={{ left: `${percentage}%` }}></div>
            </div>
            <div className="metric-slider-labels">
                <span>{formatVal(minVal)}</span>
                <span>{formatVal(maxVal)}</span>
            </div>
        </div>
    );
};

// ==========================================================================
// RECOMMENDATION CARD — PREMIUM LAYOUT
// ==========================================================================
const RecommendationCard = ({ rec, onBook }) => {
    const rankClass = rec.rank <= 3 ? `rank-${rec.rank}` : '';
    const rankEmoji = rec.rank === 1 ? "🥇" : rec.rank === 2 ? "🥈" : rec.rank === 3 ? "🥉" : `#${rec.rank}`;

    return (
        <div className="rec-card">
            <div className="rec-card-header">
                <div className="rec-card-title-group">
                    <div className={`rec-rank-badge ${rankClass}`}>{rankEmoji}</div>
                    <div>
                        <h3 className="rec-name">{rec.restaurant_name}</h3>
                        <div className="rec-meta-tags">
                            <span className="rec-cuisine-pill">{rec.cuisine}</span>
                            {rec.restaurant_type && (
                                <span className="rec-type-badge">
                                    <Icon name="store" style={{ width: '11px', height: '11px' }} />
                                    {rec.restaurant_type}
                                </span>
                            )}
                            {rec.online_order && (
                                <span className="rec-online-badge">
                                    <Icon name="wifi" style={{ width: '11px', height: '11px' }} />
                                    Online Order
                                </span>
                            )}
                        </div>
                    </div>
                </div>
                <div className="rec-rating-bubble">
                    <span>⭐</span>
                    <span>{rec.rating.toFixed(1)}</span>
                </div>
            </div>

            {/* Red Horizontal Indicator Lines for Key Sections */}
            <div className="rec-metrics-container">
                <MetricSlider 
                    label="Rating" 
                    value={rec.rating} 
                    minVal={3.0} 
                    maxVal={5.0} 
                    formatVal={v => `${v.toFixed(1)} ⭐`}
                />
                <MetricSlider 
                    label="Cost for Two" 
                    value={rec.cost_value} 
                    minVal={100} 
                    maxVal={3000} 
                    formatVal={v => `₹${v}`}
                />
                <MetricSlider 
                    label="Popularity" 
                    value={rec.votes} 
                    minVal={0} 
                    maxVal={5000} 
                    formatVal={v => v >= 5000 ? '5,000+' : v.toLocaleString()}
                />
            </div>

            {/* AI Recommendation Reason */}
            <div>
                <h5 className="rec-explanation-title">
                    <Icon name="sparkles" style={{ width: '16px', height: '16px', color: 'var(--color-primary-light)' }} />
                    AI Concierge Insight
                </h5>
                <p className="rec-explanation-text">{rec.explanation}</p>
            </div>

            {/* Actions */}
            <div className="rec-card-actions">
                {rec.book_table && (
                    <button className="action-btn btn-primary" onClick={() => onBook(rec.restaurant_name)}>
                        <Icon name="calendar" style={{ width: '16px', height: '16px' }} />
                        Book Table
                    </button>
                )}
                <button className="action-btn" onClick={() => window.open(`https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(rec.restaurant_name + ' ' + rec.cuisine)}`, '_blank')}>
                    <Icon name="navigation" style={{ width: '16px', height: '16px' }} />
                    Get Directions
                </button>
            </div>
        </div>
    );
};

// ==========================================================================
// MAIN APP ROOT
// ==========================================================================
const App = () => {
    const [locations, setLocations] = useState([]);
    const [cuisines, setCuisines] = useState([]);
    
    const [location, setLocation] = useState("");
    const [budget, setBudget] = useState(1000);
    const [partySize, setPartySize] = useState(2);
    const [cuisine, setCuisine] = useState("");
    const [minRating, setMinRating] = useState(3.5);
    const [preferences, setPreferences] = useState([]);

    // GPS & Loading states
    const [gpsLoading, setGpsLoading] = useState(false);
    const [gpsError, setGpsError] = useState("");
    const [searching, setSearching] = useState(false);

    // Results states
    const [results, setResults] = useState(null);
    const [warnings, setWarnings] = useState([]);
    const [errorMsg, setErrorMsg] = useState("");
    const [searched, setSearched] = useState(false);

    // Booking modal state
    const [bookingOpen, setBookingOpen] = useState(false);
    const [bookingRestaurant, setBookingRestaurant] = useState("");
    const [bookingSuccess, setBookingSuccess] = useState(false);
    const [bookingCode, setBookingCode] = useState("");

    // Load initial dropdown options & trigger geolocation automatically
    useEffect(() => {
        fetch("/api/locations")
            .then(res => res.json())
            .then(data => {
                setLocations(data);
                if (data && data.length > 0) {
                    setLocation(data[0]);
                    // Auto Geolocation on launch
                    detectLocation(data);
                }
            })
            .catch(err => console.error("Error loading locations:", err));

        fetch("/api/cuisines")
            .then(res => res.json())
            .then(data => setCuisines(data))
            .catch(err => console.error("Error loading cuisines:", err));
    }, []);

    // Detect GPS location and map to the closest location
    const detectLocation = (locationsList) => {
        if (!navigator.geolocation) return;
        setGpsLoading(true);
        setGpsError("");

        navigator.geolocation.getCurrentPosition(
            (position) => {
                const { latitude, longitude } = position.coords;
                const closest = getClosestLocation(latitude, longitude, locationsList || locations);
                setLocation(closest);
                setGpsLoading(false);
            },
            (error) => {
                console.warn("Auto GPS failed, user can select manually:", error);
                setGpsLoading(false);
            },
            { enableHighAccuracy: true, timeout: 8000 }
        );
    };

    const handleDetectLocation = () => {
        detectLocation(locations);
    };

    // Toggle multi-select preference tags
    const handlePrefToggle = (pref) => {
        if (preferences.includes(pref)) {
            setPreferences(preferences.filter(p => p !== pref));
        } else {
            setPreferences([...preferences, pref]);
        }
    };

    // Submit request to backend
    const handleSearch = (e) => {
        if (e) e.preventDefault();
        
        setSearching(true);
        setErrorMsg("");
        setWarnings([]);
        setResults(null);

        const payload = {
            location: location,
            budget: parseInt(budget) || null,
            party_size: partySize,
            cuisine: cuisine || null,
            min_rating: parseFloat(minRating),
            preferences: preferences
        };

        fetch("/api/recommend", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        })
        .then(res => {
            if (!res.ok) {
                return res.json().then(err => { throw new Error(err.detail || "Server error") });
            }
            return res.json();
        })
        .then(data => {
            if (data.success) {
                setResults(data);
                setWarnings(data.warnings || []);
            } else {
                setErrorMsg(data.message || "No results found matching your criteria.");
                setWarnings(data.warnings || []);
            }
            setSearching(false);
            setSearched(true);
        })
        .catch(err => {
            console.error("Search error:", err);
            setErrorMsg(err.message || "Failed to contact recommendation engine.");
            setSearching(false);
        });
    };

    // Trigger mock table booking
    const handleBookTable = (restaurantName) => {
        setBookingRestaurant(restaurantName);
        setBookingOpen(true);
        setBookingSuccess(false);

        setTimeout(() => {
            const randomCode = "TLB-" + Math.floor(1000 + Math.random() * 9000);
            setBookingCode(randomCode);
            setBookingSuccess(true);
        }, 1500);
    };

    // Compute slider background fill for budget
    const budgetPercent = ((budget - 100) / 2900) * 100;
    const partySizePercent = ((partySize - 1) / 19) * 100;
    const ratingPercent = (minRating / 5.0) * 100;

    return (
        <div className="app-container">
            {/* Header */}
            <header className="app-header">
                <div className="brand-badge">
                    <Icon name="utensils" style={{ width: '13px', height: '13px' }} />
                    The Last Bite · Café & Restaurant
                </div>
                <h1 className="app-title">AI Dining <span>Concierge</span></h1>
                <p className="app-subtitle">Premium restaurant discovery powered by artificial intelligence. Personalized picks, instant bookings, curated just for you.</p>
            </header>

            {/* Layout Grid */}
            <div className="app-grid">
                {/* Left Side: Interactive Preferences Form */}
                <aside className="sidebar-panel">
                    <h3 className="panel-title">
                        <Icon name="sliders" style={{ color: 'var(--color-primary-light)', width: '18px', height: '18px' }} />
                        Your Preferences
                    </h3>
                    <form onSubmit={handleSearch}>
                        {/* Location */}
                        <div className="form-group">
                            <label className="form-label">
                                📍 Location
                                <button type="button" className="detect-loc-btn" onClick={handleDetectLocation} disabled={gpsLoading}>
                                    <Icon name="compass" style={{ width: '11px', height: '11px' }} />
                                    {gpsLoading ? "Detecting..." : "GPS Detect"}
                                </button>
                            </label>
                            <select className="form-select" value={location} onChange={e => setLocation(e.target.value)}>
                                {locations.map(loc => (
                                    <option key={loc} value={loc}>{loc}</option>
                                ))}
                            </select>
                            {gpsError && <span style={{ fontSize: '0.75rem', color: '#f59e0b', marginTop: '4px', display: 'block' }}>{gpsError}</span>}
                        </div>

                        {/* Budget (Slider Only — No +/- buttons) */}
                        <div className="form-group">
                            <label className="form-label">
                                💰 Budget (Total)
                                <span className="label-value">₹{budget}</span>
                            </label>
                            <input 
                                type="range" 
                                className="form-range" 
                                min="100" 
                                max="3000" 
                                step="50" 
                                value={budget} 
                                onChange={e => setBudget(parseInt(e.target.value))} 
                                style={{ background: `linear-gradient(to right, var(--color-primary) ${budgetPercent}%, #1a1d2e ${budgetPercent}%)` }}
                            />
                            
                            {/* Quick Select Budget Chips */}
                            <div className="budget-chips">
                                <div className={`budget-chip ${budget === 300 ? 'active' : ''}`} onClick={() => setBudget(300)}>Budget ~₹300</div>
                                <div className={`budget-chip ${budget === 800 ? 'active' : ''}`} onClick={() => setBudget(800)}>Medium ~₹800</div>
                                <div className={`budget-chip ${budget === 1500 ? 'active' : ''}`} onClick={() => setBudget(1500)}>Premium ~₹1500</div>
                                <div className={`budget-chip ${budget === 2500 ? 'active' : ''}`} onClick={() => setBudget(2500)}>Luxury ~₹2500</div>
                            </div>
                        </div>

                        {/* Party Size Slider */}
                        <div className="form-group">
                            <label className="form-label">
                                👥 Party Size
                                <span className="label-value">{partySize} {partySize === 1 ? 'person' : 'people'}</span>
                            </label>
                            <input 
                                type="range" 
                                className="form-range" 
                                min="1" 
                                max="20" 
                                value={partySize} 
                                onChange={e => setPartySize(parseInt(e.target.value))} 
                                style={{ background: `linear-gradient(to right, var(--color-primary) ${partySizePercent}%, #1a1d2e ${partySizePercent}%)` }}
                            />
                        </div>

                        {/* Cuisine */}
                        <div className="form-group">
                            <label className="form-label">🍕 Cuisine Preference</label>
                            <select className="form-select" value={cuisine} onChange={e => setCuisine(e.target.value)}>
                                <option value="">Any (No preference)</option>
                                {cuisines.map(c => (
                                    <option key={c} value={c}>{c}</option>
                                ))}
                            </select>
                        </div>

                        {/* Rating Slider (Smooth 0.1 Steps) */}
                        <div className="form-group">
                            <label className="form-label">
                                ⭐ Minimum Rating
                                <span className="label-value" style={{ color: 'var(--color-amber)' }}>{minRating.toFixed(1)}</span>
                            </label>
                            <input 
                                type="range" 
                                className="form-range" 
                                min="0.0" 
                                max="5.0" 
                                step="0.1" 
                                value={minRating} 
                                onChange={e => setMinRating(parseFloat(e.target.value))} 
                                style={{ background: `linear-gradient(to right, var(--color-amber) ${ratingPercent}%, #1a1d2e ${ratingPercent}%)` }}
                            />
                            {/* Visual Star Rating Bar */}
                            <div className="star-rating-visual">
                                <div className="star-bar-container">
                                    <div className="star-bar-fill" style={{ width: `${ratingPercent}%` }}></div>
                                </div>
                                <div className="star-value-badge">{minRating.toFixed(1)} ⭐</div>
                            </div>
                        </div>

                        {/* Preferences */}
                        <div className="form-group">
                            <label className="form-label">🏷️ Dining Tags</label>
                            <div className="preferences-grid">
                                {VALID_PREFERENCES.map(pref => {
                                    const active = preferences.includes(pref);
                                    return (
                                        <div 
                                            key={pref} 
                                            className={`pref-checkbox-label ${active ? 'active' : ''}`}
                                            onClick={() => handlePrefToggle(pref)}
                                        >
                                            <input type="checkbox" checked={active} readOnly />
                                            <span>{pref}</span>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>

                        {/* Search Button */}
                        <button type="submit" className="search-btn" disabled={searching}>
                            <Icon name="search" style={{ width: '18px', height: '18px' }} />
                            {searching ? "Finding Tables..." : "Get Recommendations"}
                        </button>
                    </form>
                </aside>

                {/* Right Side: Dynamic Content Display */}
                <main className="results-panel">
                    {/* Warnings (if any) */}
                    {warnings.length > 0 && (
                        <div className="warnings-card">
                            <Icon name="alertTriangle" style={{ width: '18px', height: '18px', flexShrink: 0 }} />
                            <div>
                                {warnings.map((w, idx) => (
                                    <p key={idx}>{w}</p>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Searching State: Skeleton Screen Loader */}
                    {searching && (
                        <div className="loading-screen">
                            <div className="loader-animation">
                                <div className="loader-ring"></div>
                                <div className="loader-sparkle">
                                    <Icon name="sparkles" style={{ width: '36px', height: '36px', color: 'var(--color-primary)' }} />
                                </div>
                            </div>
                            <div className="loading-text">Analyzing Dining Venues...</div>
                            <p className="loading-subtext">Our AI concierge is filtering 50,000+ tables, matching budgets, checking reviews, and compiling reasons...</p>
                            
                            {/* Skeleton Cards */}
                            <div style={{ width: '100%', display: 'flex', flexDirection: 'column', gap: '1.5rem', marginTop: '2rem' }}>
                                {[1, 2].map(i => (
                                    <div key={i} className="skeleton-card">
                                        <div className="skeleton-header">
                                            <div className="skeleton-pulse skeleton-title"></div>
                                            <div className="skeleton-pulse skeleton-badge"></div>
                                        </div>
                                        <div className="skeleton-pulse skeleton-subtitle"></div>
                                        <div className="skeleton-pulse skeleton-metric-block"></div>
                                        <div className="skeleton-pulse skeleton-text-line"></div>
                                        <div className="skeleton-pulse skeleton-text-line short"></div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Error message */}
                    {errorMsg && !searching && (
                        <div className="empty-state">
                            <div className="empty-icon">😕</div>
                            <h3>No Restaurants Found</h3>
                            <p>{errorMsg}</p>
                            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
                                Try adjusting your sliders: broaden cuisines, relax budget constraints, or choose another area.
                            </p>
                        </div>
                    )}

                    {/* Initial State (Not Searched Yet) — Enhanced Welcome */}
                    {!searched && !searching && (
                        <div className="empty-state">
                            <div className="empty-icon">🍽️</div>
                            <h3>Welcome to The Last Bite</h3>
                            <p>
                                Your AI-powered dining concierge. We detect your location automatically, analyze thousands of restaurants, and curate the perfect picks just for you.
                            </p>
                            <div className="welcome-features">
                                <div className="welcome-feature-chip">
                                    <Icon name="compass" style={{ width: '14px', height: '14px', color: 'var(--color-primary-light)' }} />
                                    GPS Location
                                </div>
                                <div className="welcome-feature-chip">
                                    <Icon name="brain" style={{ width: '14px', height: '14px', color: 'var(--color-primary-light)' }} />
                                    AI Reasoning
                                </div>
                                <div className="welcome-feature-chip">
                                    <Icon name="map" style={{ width: '14px', height: '14px', color: 'var(--color-primary-light)' }} />
                                    Map View
                                </div>
                                <div className="welcome-feature-chip">
                                    <Icon name="calendar" style={{ width: '14px', height: '14px', color: 'var(--color-primary-light)' }} />
                                    Instant Booking
                                </div>
                                <div className="welcome-feature-chip">
                                    <Icon name="zap" style={{ width: '14px', height: '14px', color: 'var(--color-primary-light)' }} />
                                    Fast Results
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Recommendations Render */}
                    {results && !searching && (
                        <>
                            {/* AI Summary card */}
                            <div className="ai-summary-card">
                                <h4 className="summary-title">
                                    <Icon name="brain" style={{ width: '20px', height: '20px' }} />
                                    AI Concierge Summary
                                </h4>
                                <p className="summary-body">{results.summary}</p>
                                <div className="summary-meta">
                                    <span>
                                        <Icon name="activity" style={{ width: '13px', height: '13px' }} />
                                        {results.candidates_found} screened
                                    </span>
                                    <span>
                                        <Icon name="cpu" style={{ width: '13px', height: '13px' }} />
                                        {results.model_used}
                                    </span>
                                    <span>
                                        <Icon name="zap" style={{ width: '13px', height: '13px' }} />
                                        {results.response_time_ms}ms
                                    </span>
                                </div>
                            </div>

                            {/* Leaflet Map Integration */}
                            <MapView location={location} recommendations={results.recommendations} />

                            {/* Recommendation Cards */}
                            <div className="recommendations-grid">
                                {results.recommendations.map(rec => (
                                    <RecommendationCard 
                                        key={rec.rank} 
                                        rec={rec} 
                                        onBook={handleBookTable} 
                                    />
                                ))}
                            </div>
                        </>
                    )}
                </main>
            </div>

            {/* Success Booking Modal Backdrop */}
            <div className={`modal-overlay ${bookingOpen ? 'open' : ''}`}>
                <div className="success-modal">
                    {!bookingSuccess ? (
                        <>
                            <div className="loader-animation" style={{ margin: '0 auto 1.5rem auto' }}>
                                <div className="loader-ring"></div>
                                <div className="loader-sparkle">
                                    <Icon name="chefHat" style={{ width: '32px', height: '32px', color: 'var(--color-primary)' }} />
                                </div>
                            </div>
                            <h3 className="modal-title" style={{ fontSize: '1.5rem' }}>Securing Your Table</h3>
                            <p className="modal-text">Contacting hostess desk at {bookingRestaurant}...</p>
                        </>
                    ) : (
                        <>
                            <div className="success-circle-container">
                                <div className="success-circle">
                                    <Icon name="check" style={{ width: '40px', height: '40px', color: 'var(--color-emerald)' }} />
                                </div>
                            </div>
                            <h3 className="modal-title">Table Confirmed!</h3>
                            <p className="modal-text">Your reservation at <strong>{bookingRestaurant}</strong> has been successfully booked via <em>The Last Bite Concierge</em>.</p>
                            
                            <div className="modal-details">
                                <div className="detail-row">
                                    <span className="label">Reservation Code:</span>
                                    <span className="val" style={{ color: 'var(--color-primary-light)', fontFamily: 'monospace', fontSize: '1rem', letterSpacing: '0.5px' }}>{bookingCode}</span>
                                </div>
                                <div className="detail-row">
                                    <span className="label">Guests:</span>
                                    <span className="val">{partySize} People</span>
                                </div>
                                <div className="detail-row">
                                    <span className="label">Time:</span>
                                    <span className="val">Tonight, 8:00 PM</span>
                                </div>
                                <div className="detail-row">
                                    <span className="label">Location:</span>
                                    <span className="val">{location}</span>
                                </div>
                            </div>

                            <button className="action-btn btn-primary" style={{ width: '100%' }} onClick={() => setBookingOpen(false)}>
                                Excellent, See You There!
                            </button>
                        </>
                    )}
                </div>
            </div>

            {/* Footer */}
            <footer className="app-footer">
                <p>© 2026 The Last Bite — Café & Restaurant. Powered by AI Dining Intelligence.</p>
            </footer>
        </div>
    );
};

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
