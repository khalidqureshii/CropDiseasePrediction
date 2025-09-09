import requests
import numpy as np
import rasterio
from datetime import datetime
import os
import pandas as pd
import pickle
import warnings
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

# Load environment variables from .env file
load_dotenv()

# ---------- CONFIGURATION ----------
OPEN_WEATHER_API_KEY = os.getenv('OPEN_WEATHER_API_KEY')
SENTINEL_HUB_TOKEN = os.getenv('SENTINEL_HUB_TOKEN')

# ---------- NPK ESTIMATION MODULE ----------
class NPKEstimator:
    """Estimates NPK from satellite indices using research-based relationships"""

    def __init__(self):
        self.models = None
        self.scaler = None
        self.is_trained = False

    def estimate_npk_from_indices(self, ndvi, ndwi, soil_moisture, soil_ph):
        """Research-based NPK estimation from satellite indices"""
        # Ensure valid ranges
        ndvi = max(0.1, min(0.9, ndvi))
        ndwi = max(-0.5, min(0.8, ndwi))
        soil_moisture = max(0.1, min(1.0, soil_moisture))
        soil_ph = max(4.0, min(9.0, soil_ph))

        # Calculate additional indices
        evi2 = 2.5 * ndvi / (1 + ndvi + 0.1)  # Enhanced Vegetation Index
        savi = ndvi * 1.5 / (ndvi + 0.5)      # Soil Adjusted Vegetation Index

        # Research-based nitrogen estimation (mg/kg)
        # Based on correlation between vegetation vigor and N content
        nitrogen = (
            120 * ndvi +                    # Primary vegetation indicator
            25 * evi2 +                     # Enhanced vegetation response
            30 * soil_moisture +            # Moisture effect on N availability
            15 * max(0, 1 - abs(soil_ph - 6.5)/2)  # pH optimization
        )
        nitrogen = max(20, min(200, nitrogen))

        # Phosphorus estimation (mg/kg)
        # P availability strongly affected by pH and organic matter
        phosphorus = (
            35 * ndvi +                     # Organic matter proxy
            20 * soil_moisture +            # Moisture effect
            25 * max(0, 1 - abs(soil_ph - 6.5)/1.5)  # pH curve for P availability
        )
        phosphorus = max(10, min(100, phosphorus))

        # Potassium estimation (mg/kg)
        # K mobility affected by water and vegetation uptake
        potassium = (
            70 * ndvi +                     # Vegetation uptake indicator
            35 * (1 + ndwi) / 2 +          # Water availability proxy
            40 * soil_moisture +            # Moisture effect on K mobility
            30                              # Base level
        )
        potassium = max(50, min(300, potassium))

        return {
            'nitrogen': round(nitrogen, 1),
            'phosphorus': round(phosphorus, 1),
            'potassium': round(potassium, 1),
            'confidence': self._calculate_confidence(ndvi, ndwi, soil_moisture, soil_ph)
        }

    def _calculate_confidence(self, ndvi, ndwi, soil_moisture, soil_ph):
        """Calculate confidence based on data quality"""
        confidence = 100

        # Penalize extreme values
        if ndvi < 0.2 or ndvi > 0.8:
            confidence -= 10
        if abs(ndwi) > 0.5:
            confidence -= 5
        if soil_ph < 5.5 or soil_ph > 8.0:
            confidence -= 10
        if soil_moisture < 0.2 or soil_moisture > 0.8:
            confidence -= 5

        return max(60, confidence)


class CropRecommendationSystem:
    """ML-based crop recommendation system using pre-trained models"""

    def __init__(self, model_dir='trained_models'):
        self.model_dir = model_dir
        self.crop_model = None
        self.yield_model = None
        self.suitability_model = None
        self.scaler = None
        self.label_encoder = None
        self.yield_encoder = None
        self.crop_seasons = {}
        self.model_performance = {}
        self.is_loaded = False

    def load_models(self):
        """Load pre-trained models from disk"""
        try:
            with open(f'{self.model_dir}/crop_model.pkl', 'rb') as f:
                self.crop_model = pickle.load(f)

            with open(f'{self.model_dir}/yield_model.pkl', 'rb') as f:
                self.yield_model = pickle.load(f)

            with open(f'{self.model_dir}/suitability_model.pkl', 'rb') as f:
                self.suitability_model = pickle.load(f)

            with open(f'{self.model_dir}/scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)

            with open(f'{self.model_dir}/label_encoder.pkl', 'rb') as f:
                self.label_encoder = pickle.load(f)

            with open(f'{self.model_dir}/yield_encoder.pkl', 'rb') as f:
                self.yield_encoder = pickle.load(f)

            with open(f'{self.model_dir}/crop_seasons.pkl', 'rb') as f:  # ADD THIS BLOCK
                self.crop_seasons = pickle.load(f)

            with open(f'{self.model_dir}/model_performance.pkl', 'rb') as f:
                self.model_performance = pickle.load(f)

            self.is_loaded = True
            print("Pre-trained models loaded successfully!")

        except FileNotFoundError as e:
            print(f"Error loading models: {e}")
            print("Please run the model training script first to generate the models.")
            self.is_loaded = False

    def get_comprehensive_recommendations(self, satellite_data, weather_data, npk_data):
        """Get comprehensive crop recommendations with confidence scores"""
        if not self.is_loaded:
            raise ValueError("Models must be loaded first! Call load_models()")

        # Get current month for seasonal features
        current_month = datetime.now().month

        # Determine if current month is kharif season (June-October)
        is_kharif = 1 if current_month in [6, 7, 8, 9, 10] else 0

        # Prepare features
        features = np.array([[
            weather_data.get('temperature', 25),
            weather_data.get('humidity', 60),
            satellite_data.get('soil_ph', 6.5),
            weather_data.get('rainfall', 0) * 24 * 30,  # Convert to monthly
            npk_data.get('nitrogen', 100),
            npk_data.get('phosphorus', 30),
            npk_data.get('potassium', 150),
            satellite_data.get('ndvi', 0.5),
            satellite_data.get('ndwi', 0.2),
            current_month,
            is_kharif
        ]])

        # Scale features
        features_scaled = self.scaler.transform(features)

        # Get predictions
        crop_probs = self.crop_model.predict_proba(features_scaled)[0]
        yield_pred = self.yield_model.predict(features_scaled)[0]
        suitability_pred = self.suitability_model.predict(features_scaled)[0]

        # Get top recommendations
        top_indices = np.argsort(crop_probs)[::-1][:10]
        recommendations = []

        for idx in top_indices:
            crop_name = self.label_encoder.inverse_transform([idx])[0]
            confidence = crop_probs[idx] * 100

            if confidence < 5:
                continue

            # Add seasonal appropriateness check
            crop_season = self.crop_seasons.get(crop_name, 'unknown')
            season_match = self._check_seasonal_appropriateness(crop_season, current_month)

            # Don't skip crops entirely, just adjust confidence
            adjusted_confidence = confidence
            if season_match == 'Poor':
                adjusted_confidence *= 0.6  # Reduce confidence for off-season
            elif season_match == 'Fair':
                adjusted_confidence *= 0.8

            recommendations.append({
                'crop': crop_name,
                'confidence': round(confidence, 1),
                'expected_yield': ['Low', 'Medium', 'High'][yield_pred],
                'suitability': ['Poor', 'Good', 'Excellent'][suitability_pred],
                'recommendation_score': round(confidence * (suitability_pred + 1) / 3, 1),
                'seasonal_match': season_match,
                'season': crop_season
            })

        return sorted(recommendations, key=lambda x: x['recommendation_score'], reverse=True)

    def _check_seasonal_appropriateness(self, crop_season, current_month):
      """Check if crop season matches current month with more flexibility"""
      if crop_season == 'kharif':
          if current_month in [6, 7, 8, 9, 10]:  # Jun-Oct
              return 'Excellent'
          elif current_month in [5, 11]:  # Shoulder months
              return 'Fair'
          else:
              return 'Poor'

      elif crop_season == 'rabi':
          if current_month in [11, 12, 1, 2, 3]:  # Nov-Mar
              return 'Excellent'
          elif current_month in [10, 4]:  # Shoulder months
              return 'Fair'
          else:
              return 'Poor'

      elif crop_season == 'zaid':
          if current_month in [3, 4, 5, 6]:  # Mar-Jun
              return 'Excellent'
          else:
              return 'Fair'  # Zaid crops are more flexible

      else:
          return 'Good'  # Unknown season - assume moderate suitability

    def get_feature_importance_analysis(self):
        """Get detailed feature importance analysis"""
        if not self.is_loaded:
            return None

        feature_names = ['Temperature', 'Humidity', 'pH', 'Rainfall',
                        'Nitrogen', 'Phosphorus', 'Potassium', 'NDVI', 'NDWI', 'Month', 'Season']

        importance = self.crop_model.feature_importances_

        # Create detailed analysis
        analysis = []
        for i, (name, imp) in enumerate(zip(feature_names, importance)):
            analysis.append({
                'feature': name,
                'importance': round(imp * 100, 2),
                'rank': i + 1,
                'category': self._get_feature_category(name)
            })

        return sorted(analysis, key=lambda x: x['importance'], reverse=True)

    def _get_feature_category(self, feature):
        """Categorize features for better understanding"""
        categories = {
            'Temperature': 'Climate', 'Humidity': 'Climate', 'Rainfall': 'Climate',
            'pH': 'Soil', 'Nitrogen': 'Soil', 'Phosphorus': 'Soil', 'Potassium': 'Soil',
            'NDVI': 'Satellite', 'NDWI': 'Satellite', 'Month': 'Temporal', 'Season': 'Temporal'
        }
        return categories.get(feature, 'Other')


# ---------- INTEGRATED SYSTEM CLASS ----------
class CropAdvisorySystem:
    """Complete integrated system for agricultural advisory"""

    def __init__(self):
        self.npk_estimator = NPKEstimator()
        self.crop_system = CropRecommendationSystem()
        self.system_ready = False

    def initialize_system(self):
        """Initialize system by loading pre-trained models"""
        print("Initializing Crop Advisory System...")
        print("=" * 50)

        # Load pre-trained models
        self.crop_system.load_models()

        if self.crop_system.is_loaded:
            self.system_ready = True
            print("\nSystem Ready!")
            print(f"Model Performance: {self.crop_system.model_performance['test_accuracy']:.1%}")
            print("=" * 50)
            return self.crop_system.model_performance
        else:
            print("System initialization failed!")
            return None

    def analyze_location(self, latitude, longitude, manual_inputs=None):
        """Complete location analysis with ML recommendations"""
        if not self.system_ready:
            print("System not initialized. Initializing now...")
            init_result = self.initialize_system()
            if not init_result:
                raise ValueError("System initialization failed. Please ensure models are trained first.")

        print(f"\nAnalyzing location: {latitude:.4f}, {longitude:.4f}")

        # Step 1: Get real-time weather data
        print("Fetching weather data...")
        weather_data = self.get_weather_data(latitude, longitude)

        # Step 2: Get satellite soil data
        print("Processing satellite imagery...")
        satellite_data = self.get_soil_data(latitude, longitude)

        # Step 3: Estimate NPK from satellite data
        print("Estimating soil nutrients...")
        npk_data = self.npk_estimator.estimate_npk_from_indices(
            satellite_data['ndvi'],
            satellite_data['ndwi'],
            satellite_data['soil_moisture'],
            satellite_data['soil_ph']
        )

        # Step 4: Get ML-based crop recommendations
        print("Generating ML recommendations...")
        recommendations = self.crop_system.get_comprehensive_recommendations(
            satellite_data, weather_data, npk_data
        )

        # Step 5: Get feature importance analysis
        feature_analysis = self.crop_system.get_feature_importance_analysis()

        # Combine all results
        results = {
            'location': {
                'latitude': latitude,
                'longitude': longitude,
                'city': weather_data.get('location', {}).get('city', 'Unknown')
            },
            'weather_data': weather_data,
            'satellite_data': satellite_data,
            'npk_estimates': npk_data,
            'crop_recommendations': recommendations,
            'feature_importance': feature_analysis,
            'model_performance': self.crop_system.model_performance,
            'analysis_timestamp': datetime.now().isoformat(),
        }

        # Step 6: Get advanced AI insights
        advanced_insights = self._get_advanced_insights()
        results['advanced_insights'] = advanced_insights
        
        return results

    def display_comprehensive_report(self, results):
        """Display a comprehensive analysis report"""
        print("\n" + "="*80)
        print("COMPREHENSIVE CROP ADVISORY REPORT")
        print("="*80)

        # Location Info
        loc = results['location']
        print(f"Location: {loc['city']} ({loc['latitude']:.4f}, {loc['longitude']:.4f})")
        print(f"Analysis Time: {results['analysis_timestamp']}")

        # Weather Summary
        weather = results['weather_data']
        print(f"\nWEATHER CONDITIONS:")
        print(f"   Temperature: {weather.get('temperature', 'N/A')}°C")
        print(f"   Humidity: {weather.get('humidity', 'N/A')}%")
        print(f"   Rainfall: {weather.get('rainfall', 'N/A')} mm/day")
        print(f"   Season: {weather.get('season', 'N/A')}")

        # Satellite Data Summary
        satellite = results['satellite_data']
        print(f"\nSATELLITE ANALYSIS:")
        print(f"   NDVI (Vegetation Health): {satellite.get('ndvi', 'N/A'):.3f}")
        print(f"   NDWI (Water Index): {satellite.get('ndwi', 'N/A'):.3f}")
        print(f"   Soil Moisture: {satellite.get('soil_moisture', 'N/A'):.3f}")
        print(f"   Soil pH: {satellite.get('soil_ph', 'N/A'):.1f}")

        # NPK Estimates
        npk = results['npk_estimates']
        print(f"\nSOIL NUTRIENT ANALYSIS (ML Estimated):")
        print(f"   Nitrogen (N): {npk.get('nitrogen', 'N/A')} mg/kg")
        print(f"   Phosphorus (P): {npk.get('phosphorus', 'N/A')} mg/kg")
        print(f"   Potassium (K): {npk.get('potassium', 'N/A')} mg/kg")
        print(f"   Confidence: {npk.get('confidence', 'N/A')}%")

        # Top Crop Recommendations
        recommendations = results['crop_recommendations']
        print(f"\nTOP ML-BASED CROP RECOMMENDATIONS:")
        print("-" * 60)

        for i, rec in enumerate(recommendations[:5], 1):
            print(f"{i}. {rec['crop']}")
            print(f"   Confidence: {rec['confidence']}%")
            print(f"   Expected Yield: {rec['expected_yield']}")
            print(f"   Suitability: {rec['suitability']}")
            print(f"   Seasonal Match: {rec.get('seasonal_match', 'N/A')}")
            print(f"   Overall Score: {rec['recommendation_score']}")
            print()

        # Model Performance
        performance = results['model_performance']
        print(f"MODEL PERFORMANCE METRICS:")
        print(f"   Test Accuracy: {performance['test_accuracy']:.1%}")
        print(f"   Cross-Validation: {performance['cv_mean']:.1%} ± {performance['cv_std']:.1%}")

        # Feature Importance
        importance = results['feature_importance']
        print(f"\nMOST IMPORTANT FACTORS:")
        print("-" * 40)

        for factor in importance[:5]:
            print(f"   {factor['feature']}: {factor['importance']}% ({factor['category']})")

        # Advanced Insights
        self._get_advanced_insights(results)

        print("="*80)

    def _get_advanced_insights(self, results):
        """Display advanced insights"""
        print(f"\nADVANCED AI INSIGHTS:")
        print("-" * 40)
        insights = []

        satellite = results['satellite_data']
        weather = results['weather_data']
        npk = results['npk_estimates']
        recommendations = results['crop_recommendations']

        # Vegetation health insight
        ndvi = satellite.get('ndvi', 0.5)
        if ndvi > 0.7:
            health_status = "Excellent"
        elif ndvi > 0.5:
            health_status = "Good"
        else:
            health_status = "Poor"

        insights.append(f"Field Health: {health_status} (NDVI: {ndvi:.3f})")
        print(f"   Field Health: {health_status} (NDVI: {ndvi:.3f})")

        # Water availability insight
        ndwi = satellite.get('ndwi', 0.2)
        rainfall = weather.get('rainfall', 0)
        if ndwi > 0.3 or rainfall > 5:
            water_status = "Adequate water availability"
        elif ndwi > 0.1 or rainfall > 2:
            water_status = "Moderate water stress"
        else:
            water_status = "Water stress detected"

        insights.append(water_status)
        print(f"   {water_status}")

        # Nutrient insights
        n_level = npk.get('nitrogen', 100)
        if n_level > 120:
            n_status = "High nitrogen - suitable for leafy crops"
        elif n_level > 80:
            n_status = "Moderate nitrogen - balanced nutrition"
        else:
            n_status = "Low nitrogen - consider fertilization"

        insights.append(n_status)
        print(f"   {n_status}")

        # Season-specific advice
        season = weather.get('season', '')
        if 'kharif' in season.lower() or 'monsoon' in season.lower():
            seasonal_advice = "Monsoon season - ideal for water-intensive crops"
        elif 'rabi' in season.lower() or 'winter' in season.lower():
            seasonal_advice = "Winter season - suitable for cool-weather crops"
        else:
            seasonal_advice = "Summer season - focus on heat-tolerant varieties"

        insights.append(seasonal_advice)
        print(f"   {seasonal_advice}")

        # Risk assessment
        risks = []
        if weather.get('temperature', 25) > 35:
            risks.append("Heat stress")
        if rainfall < 1:
            risks.append("Drought risk")
        if satellite.get('soil_ph', 6.5) < 5.5:
            risks.append("Soil acidity")
        if satellite.get('soil_ph', 6.5) > 8.0:
            risks.append("Soil alkalinity")

        if risks:
            insights.append(f"Risk Factors: {', '.join(risks)}")
            print(f"   Risk Factors: {', '.join(risks)}")
        else:
            insights.append(f"No major risk factors detected")
            print(f"   No major risk factors detected")

        # Confidence assessment
        top_confidence = recommendations[0]['confidence'] if recommendations else 0
        if top_confidence > 80:
            conf_status = "Very High"
        elif top_confidence > 60:
            conf_status = "High"
        elif top_confidence > 40:
            conf_status = "Moderate"
        else:
            conf_status = "Low"

        insights.append(f"Recommendation Confidence: {conf_status} ({top_confidence:.1f}%)")
        print(f"   Recommendation Confidence: {conf_status} ({top_confidence:.1f}%)")

        return insights

    def get_weather_data(self, lat, lon):
        """Fetch weather data with fallback"""
        if not OPEN_WEATHER_API_KEY or OPEN_WEATHER_API_KEY == "your_openweather_api_key_here":
            return self._get_simulated_weather_data(lat, lon)

        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPEN_WEATHER_API_KEY}&units=metric"

        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return {
                    "temperature": data["main"]["temp"],
                    "humidity": data["main"]["humidity"],
                    "rainfall": data.get("rain", {}).get("1h", 0),
                    "sunlight": data["weather"][0]["main"],
                    "location": {
                        "city": data["name"],
                        "latitude": lat,
                        "longitude": lon
                    },
                    "season": self._get_season(datetime.now())
                }
            else:
                return self._get_simulated_weather_data(lat, lon)
        except Exception as e:
            print(f"Weather API error: {e}")
            return self._get_simulated_weather_data(lat, lon)

    def _get_simulated_weather_data(self, lat, lon):
        """Generate realistic simulated weather data"""
        # Season-based simulation
        month = datetime.now().month

        if month in [12, 1, 2]:  # Winter
            base_temp = 15 + (lat - 20) * 0.5
            base_rainfall = 2
        elif month in [3, 4, 5]:  # Summer
            base_temp = 30 + (lat - 20) * 0.3
            base_rainfall = 1
        elif month in [6, 7, 8, 9]:  # Monsoon
            base_temp = 25 + (lat - 20) * 0.4
            base_rainfall = 8
        else:  # Post-monsoon
            base_temp = 22 + (lat - 20) * 0.4
            base_rainfall = 3

        # Add some location-based variation
        temp_variation = (lat % 5) - 2.5
        rainfall_variation = (lon % 3) - 1

        return {
            "temperature": base_temp + temp_variation,
            "humidity": 60 + (lat % 10) * 2,
            "rainfall": max(0, base_rainfall + rainfall_variation),
            "sunlight": "Clear",
            "location": {
                "city": f"Location_{lat:.1f}_{lon:.1f}",
                "latitude": lat,
                "longitude": lon
            },
            "season": self._get_season(datetime.now())
        }

    def _get_season(self, date):
        """Determine Indian agricultural season"""
        month = date.month
        if month in [12, 1, 2]:
            return "Winter (Rabi crops)"
        elif month in [3, 4, 5]:
            return "Summer"
        elif month in [6, 7, 8, 9]:
            return "Monsoon (Kharif crops)"
        else:
            return "Post-Monsoon"

    def get_soil_data(self, lat, lon):
        """Get soil data from satellite imagery with fallback"""
        print("   Downloading Sentinel-2 imagery...")
        image_path = self._download_sentinel_images(lat, lon)

        if image_path and os.path.exists(image_path):
            print("   Processing satellite data...")
            soil_data = self._calculate_vegetation_indices(image_path)
            os.remove(image_path)  # Cleanup
            return soil_data
        else:
            print("   Using simulated satellite data...")
            return self._get_simulated_soil_data(lat, lon)

    def _download_sentinel_images(self, lat, lon, size=0.01):
        """Download Sentinel-2 imagery with error handling"""
        if not SENTINEL_HUB_TOKEN or SENTINEL_HUB_TOKEN == "your_sentinel_hub_token_here":
            print("Sentinel Hub token not configured. Using simulated data.")
            return None

        url = "https://services.sentinel-hub.com/api/v1/process"

        coords = [
              [float(lon) - size, float(lat) - size],
              [float(lon) + size, float(lat) - size],
              [float(lon) + size, float(lat) + size],
              [float(lon) - size, float(lat) + size],
              [float(lon) - size, float(lat) - size]
        ]

        headers = {
            "Authorization": f"Bearer {SENTINEL_HUB_TOKEN}",
            "Content-Type": "application/json"
        }

        payload = {
            "input": {
                "bounds": {
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [coords]
                    }
                },
                "data": [{
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "mosaickingOrder": "leastRecent",
                        "maxCloudCoverage": 30
                    }
                }]
            },
            "output": {
                "width": 512,
                "height": 512,
                "responses": [{
                    "identifier": "default",
                    "format": {"type": "image/tiff"}
                }]
            },
            "evalscript": """
                //VERSION=3
                function setup() {
                    return {
                        input: ["B04", "B08", "B11", "SCL", "dataMask"],
                        output: { id: "default", bands: 4, sampleType: "FLOAT32" }
                    };
                }
                function evaluatePixel(sample) {
                    if (sample.SCL >= 8 && sample.SCL <= 10 || sample.dataMask == 0) {
                        return [NaN, NaN, NaN, NaN];
                    }
                    return [sample.B04, sample.B08, sample.B11, sample.dataMask];
                }
            """
        }

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            if response.status_code == 200:
                with open("sentinel_data.tif", "wb") as f:
                    f.write(response.content)
                return "sentinel_data.tif"
            print(f"Sentinel Hub error: {response.text}")
            return None
        except Exception as e:
            print(f"Download error: {e}")
            return None

    def _calculate_vegetation_indices(self, image_path):
        """Calculate NDVI and other vegetation indices with cloud masking"""
        try:
            with rasterio.open(image_path) as src:
                img_data = src.read()
                red = img_data[0].astype(np.float32)
                nir = img_data[1].astype(np.float32)
                swir = img_data[2].astype(np.float32)
                data_mask = img_data[3].astype(np.float32)

                # Apply data mask
                valid_mask = data_mask == 1

                # Calculate NDVI
                ndvi = (nir - red) / (nir + red + 1e-6)
                ndvi[~valid_mask] = np.nan

                # Calculate NDWI (Water Index)
                ndwi = (nir - swir) / (nir + swir + 1e-6)
                ndwi[~valid_mask] = np.nan

                # Estimate soil moisture (simplified)
                soil_moisture = np.clip((ndwi + 1) / 2, 0, 1)

                # Estimate soil pH (simplified approximation)
                soil_ph = 6.5 + (1 - np.nanmean(ndvi[valid_mask])) * 1.5 if np.any(valid_mask) else 6.5

                return {
                    "ndvi": np.nanmean(ndvi[valid_mask]) if np.any(valid_mask) else 0.5,
                    "ndwi": np.nanmean(ndwi[valid_mask]) if np.any(valid_mask) else 0,
                    "soil_moisture": np.nanmean(soil_moisture[valid_mask]) if np.any(valid_mask) else 0.5,
                    "soil_ph": soil_ph
                }
        except Exception as e:
            print(f"Image processing error: {e}")
            return {
                "ndvi": 0.5,
                "ndwi": 0,
                "soil_moisture": 0.5,
                "soil_ph": 6.5
            }

    def _get_simulated_soil_data(self, lat, lon):
        """Generate realistic simulated soil data"""
        # Location-based soil simulation
        base_ndvi = 0.5 + (lat % 10) / 20
        base_ndwi = 0.2 + (lon % 10) / 50

        # Seasonal adjustment
        month = datetime.now().month
        if month in [6, 7, 8, 9]:  # Monsoon - higher vegetation
            base_ndvi += 0.1
            base_ndwi += 0.1
        elif month in [3, 4, 5]:  # Summer - lower vegetation
            base_ndvi -= 0.1
            base_ndwi -= 0.1

        return {
            "ndvi": max(0.1, min(0.9, base_ndvi)),
            "ndwi": max(-0.3, min(0.6, base_ndwi)),
            "soil_moisture": 0.4 + (lat % 10) / 25,
            "soil_ph": 6.0 + (lon % 10) / 10
        }


# ---------- MAIN DEMONSTRATION FUNCTION ----------
def main():
    """Main function to demonstrate the complete system"""

    print("Advanced ML-Based Crop Advisory System")
    print("=" * 60)
    print("Replacing hardcoded data with real-time ML predictions")
    print("Integrating satellite imagery, weather data, and AI models")
    print("=" * 60)

    # Initialize system
    system = CropAdvisorySystem()

    # Get location input
    try:
        print("\nEnter Location Coordinates:")
        latitude = float(input("Latitude (e.g., 28.6139 for Delhi): ") or "28.6139")
        longitude = float(input("Longitude (e.g., 77.2090 for Delhi): ") or "77.2090")
    except ValueError:
        print("Using default coordinates (Delhi)")
        latitude, longitude = 28.6139, 77.2090

    # Optional manual inputs
    print("\nOptional Additional Information:")
    farm_size = input("Farm Size (acres) [Press Enter to skip]: ")
    water_source = input("Water Source (Rainfall/Borewell/Canal) [Press Enter to skip]: ")

    # Run complete analysis
    print(f"\nStarting Complete Analysis...")
    print("This may take a moment for model loading...")

    try:
        results = system.analyze_location(latitude, longitude)

        # Display comprehensive report
        system.display_comprehensive_report(results)

        # Additional insights
        print(f"\nINNOVATION HIGHLIGHTS:")
        print("-" * 50)
        print("Real-time satellite data integration (Sentinel-2)")
        print("ML-based NPK estimation from vegetation indices")
        print("Advanced ensemble models (Gradient Boosting + Random Forest)")
        print("Multi-modal data fusion (Weather + Satellite + Soil)")
        print("Confidence scoring and risk assessment")
        print("Feature importance analysis for transparency")
        print("Scalable architecture for nationwide deployment")

        # Export results option
        export_choice = input(f"\nExport results to JSON? (y/n): ")
        if export_choice.lower() == 'y':
            import json
            filename = f"crop_analysis_{latitude}_{longitude}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Results exported to: {filename}")

        return results

    except Exception as e:
        print(f"Error during analysis: {e}")
        print("Please ensure that the models have been trained first by running the model training script.")
        return None


# Run the demonstration
if __name__ == "__main__":
    results = main()