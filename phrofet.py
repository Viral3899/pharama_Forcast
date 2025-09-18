#!/usr/bin/env python3
"""
Standalone Prophet Model for Pharmaceutical Sales Forecasting
This version works independently without other model dependencies
"""

import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
import yaml
import argparse
import glob

# Prophet imports
try:
    import prophet
    from prophet import Prophet
    print(f"Prophet {prophet.__version__} loaded successfully")
except ImportError as e:
    print(f"Prophet import failed: {e}")
    sys.exit(1)

import matplotlib.pyplot as plt                                                         
import seaborn as sns
from datetime import datetime, timedelta
import joblib
 
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StandalonePharmaceuticalProphetModel:
    """
    Standalone Prophet model for pharmaceutical sales forecasting
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Prophet model
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model = None
        self.forecast_result = None  # Changed from self.forecast to avoid naming conflict
        self.metrics = {}
        
        # Configuration parameters with better defaults
        prophet_config = config.get('models', {}).get('prophet', {})
        self.daily_seasonality = prophet_config.get('daily_seasonality', False)  # Usually False for daily data
        self.weekly_seasonality = prophet_config.get('weekly_seasonality', True)
        self.yearly_seasonality = prophet_config.get('yearly_seasonality', True)
        self.seasonality_mode = prophet_config.get('seasonality_mode', 'additive')
        
        # Suppress Prophet logging
        logging.getLogger('prophet').setLevel(logging.ERROR)
        logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
        
        logger.info("Standalone Prophet model initialized")
    
    def prepare_data(self, data: pd.DataFrame, date_col: str, target_col: str, 
                    test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for Prophet model
        
        Args:
            data: Input dataframe
            date_col: Date column name
            target_col: Target column name
            test_size: Test set size
            
        Returns:
            Tuple of train and test dataframes
        """
        logger.info("Preparing data for Prophet model...")
        
        try:
            # Validate input data
            if data is None or data.empty:
                raise ValueError("Input data is None or empty")
            
            if date_col not in data.columns:
                raise ValueError(f"Date column '{date_col}' not found in data")
            
            if target_col not in data.columns:
                raise ValueError(f"Target column '{target_col}' not found in data")
            
            # Create Prophet format DataFrame
            prophet_df = data[[date_col, target_col]].copy()
            prophet_df.columns = ['ds', 'y']
            
            # Ensure proper data types
            prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
            prophet_df['y'] = pd.to_numeric(prophet_df['y'], errors='coerce')
            
            # Remove any null values
            initial_len = len(prophet_df)
            prophet_df = prophet_df.dropna()
            final_len = len(prophet_df)
            
            if final_len < initial_len:
                logger.warning(f"Removed {initial_len - final_len} rows with null values")
            
            if len(prophet_df) == 0:
                raise ValueError("No valid data remaining after cleaning")
            
            # Sort by date
            prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)
            
            # Validate we have enough data
            if len(prophet_df) < 10:
                raise ValueError("Insufficient data for training (need at least 10 data points)")
            
            # Split into train and test
            split_idx = int(len(prophet_df) * (1 - test_size))
            train_df = prophet_df[:split_idx].copy()
            test_df = prophet_df[split_idx:].copy()
            
            # Ensure train set has enough data
            if len(train_df) < 5:
                raise ValueError("Training set too small (need at least 5 data points)")
            
            logger.info(f"Data prepared: {len(train_df)} train, {len(test_df)} test samples")
            return train_df, test_df
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def fit_model(self, train_data: pd.DataFrame) -> bool:
        """
        Fit the Prophet model
        
        Args:
            train_data: Training data in Prophet format
            
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Fitting Prophet model...")
        
        try:
            # Validate training data
            if train_data is None or train_data.empty:
                raise ValueError("Training data is None or empty")
            
            required_cols = ['ds', 'y']
            if not all(col in train_data.columns for col in required_cols):
                raise ValueError(f"Training data must have columns: {required_cols}")
            
            # Check for sufficient data
            if len(train_data) < 5:
                raise ValueError("Insufficient training data (need at least 5 data points)")
            
            # Create Prophet model with error handling
            try:
                self.model = Prophet(
                    daily_seasonality=self.daily_seasonality,
                    weekly_seasonality=self.weekly_seasonality,
                    yearly_seasonality=self.yearly_seasonality,
                    seasonality_mode=self.seasonality_mode,
                    uncertainty_samples=100  # Reduce for faster execution
                )
                
                logger.info("Prophet model created successfully")
                
            except Exception as e:
                logger.error(f"Error creating Prophet model: {e}")
                self.model = None
                return False
            
            # Fit the model with error handling
            try:
                self.model.fit(train_data)
                logger.info("Prophet model fitted successfully")
                return True
                
            except Exception as e:
                logger.error(f"Error fitting Prophet model: {e}")
                self.model = None
                return False
            
        except Exception as e:
            logger.error(f"Error in fit_model: {e}")
            self.model = None
            return False
    
    def forecast(self, periods: int, freq: str = 'D') -> pd.DataFrame:
        """
        Generate forecast using fitted model
        
        Args:
            periods: Number of periods to forecast
            freq: Frequency of forecasting
            
        Returns:
            DataFrame with forecast results
        """
        logger.info(f"Generating forecast for {periods} periods...")
        
        try:
            # Validate model exists and is fitted
            if self.model is None:
                logger.error("Model is None. Call fit_model() first and ensure it succeeds.")
                return pd.DataFrame()
            
            # Check if model has been fitted (has history attribute)
            if not hasattr(self.model, 'history'):
                logger.error("Model has not been fitted. Call fit_model() first.")
                return pd.DataFrame()
            
            if self.model.history is None or self.model.history.empty:
                logger.error("Model history is None or empty. Model fitting may have failed.")
                return pd.DataFrame()
            
            # Validate inputs
            if periods <= 0:
                raise ValueError("Periods must be positive")
            
            # Create future dataframe
            try:
                future = self.model.make_future_dataframe(periods=periods, freq=freq)
                logger.info(f"Future dataframe created with {len(future)} rows")
                
            except Exception as e:
                logger.error(f"Error creating future dataframe: {e}")
                return pd.DataFrame()
            
            # Generate forecast
            try:
                forecast = self.model.predict(future)
                self.forecast_result = forecast  # Changed from self.forecast
                
                logger.info(f"Forecast generated: {len(forecast)} predictions")
                return forecast
                
            except Exception as e:
                logger.error(f"Error generating predictions: {e}")
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error in forecast method: {e}")
            return pd.DataFrame()
    
    def evaluate_model(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate model performance on test data
        
        Args:
            test_data: Test data in Prophet format
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating Prophet model...")
        
        try:
            # Validate model and test data
            if self.model is None:
                logger.error("Model is None. Call fit_model() first.")
                return {}
            
            if not hasattr(self.model, 'history'):
                logger.error("Model has not been fitted. Call fit_model() first.")
                return {}
            
            if test_data is None or test_data.empty:
                logger.warning("No test data available for evaluation")
                return {}
            
            # Generate predictions for test period
            future = pd.DataFrame({'ds': test_data['ds']})
            predictions = self.model.predict(future)
            
            # Calculate metrics
            y_true = test_data['y'].values
            y_pred = predictions['yhat'].values
            
            # Import sklearn metrics with error handling
            try:
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            except ImportError:
                logger.error("sklearn not available for metrics calculation")
                return {}
            
            # Basic metrics
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            # MAPE (Mean Absolute Percentage Error) with zero handling
            mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100
            
            # Directional accuracy
            if len(y_true) > 1:
                direction_true = np.diff(y_true) > 0
                direction_pred = np.diff(y_pred) > 0
                directional_accuracy = np.mean(direction_true == direction_pred) * 100
            else:
                directional_accuracy = 0.0
            
            self.metrics = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2_score': r2,
                'mape': mape,
                'directional_accuracy': directional_accuracy
            }
            
            logger.info(f"Model evaluation completed. RMSE: {rmse:.2f}, R²: {r2:.3f}")
            return self.metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {}
    
    def plot_forecast(self, historical_data: pd.DataFrame = None, 
                     save_path: str = None) -> None:
        """
        Plot forecast results
        
        Args:
            historical_data: Historical data for plotting
            save_path: Path to save the plot
        """
        logger.info("Creating forecast plot...")
        
        try:
            if self.forecast_result is None or self.forecast_result.empty:  # Changed from self.forecast
                logger.error("No forecast available. Call forecast() first.")
                return
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(15, 8))
            
            # Plot historical data if available
            if historical_data is not None and not historical_data.empty:
                ax.plot(historical_data['ds'], historical_data['y'], 
                       label='Historical Data', color='blue', linewidth=2)
            
            # Plot forecast
            forecast_data = self.forecast_result  # Changed from self.forecast
            ax.plot(forecast_data['ds'], forecast_data['yhat'], 
                   label='Forecast', color='red', linewidth=2)
            
            # Plot confidence intervals
            ax.fill_between(forecast_data['ds'], 
                           forecast_data['yhat_lower'], 
                           forecast_data['yhat_upper'], 
                           alpha=0.3, color='red', label='Confidence Interval')
            
            # Formatting
            ax.set_title('Pharmaceutical Sales Forecast (Prophet Model)', 
                        fontsize=16, fontweight='bold')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Sales Amount', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            if save_path:
                # Create directory if it doesn't exist
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error creating plot: {e}")
    
    def save_model(self, filepath: str = None) -> bool:
        """
        Save the fitted model
        
        Args:
            filepath: Path to save the model
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.model is None:
                logger.error("No model to save. Fit model first.")
                return False
            
            if filepath is None:
                filepath = "models/prophet_model.pkl"
            
            # Create directory if it doesn't exist
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            # Save model
            joblib.dump(self.model, filepath)
            logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        Load a saved model
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not Path(filepath).exists():
                logger.error(f"Model file not found: {filepath}")
                return False
            
            self.model = joblib.load(filepath)
            logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

def load_data_from_csv(data_path: str, date_col: str, target_col: str) -> Optional[pd.DataFrame]:
    """Load dataset from a CSV file or directory containing CSV files.

    If data_path is a directory, the first .csv file will be used.
    Returns a DataFrame with at least the specified date and target columns.
    """
    try:
        path_obj = Path(data_path)
        if path_obj.is_dir():
            csv_files = sorted(path_obj.glob('*.csv'))
            if not csv_files:
                logger.warning(f"No CSV files found in directory: {data_path}")
                return None
            csv_path = csv_files[0]
        else:
            csv_path = path_obj
            if not csv_path.exists():
                logger.warning(f"CSV file not found: {data_path}")
                return None

        logger.info(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)
        missing_cols = [c for c in [date_col, target_col] if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in CSV: {missing_cols}")
        return df

    except Exception as e:
        logger.error(f"Error loading CSV data: {e}")
        return None


def create_sample_data() -> pd.DataFrame:
    """Create sample pharmaceutical sales data for testing
    Generates 5 years of daily data up to today.
    """
    
    # Generate 5 years of daily sales data up to today
    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.DateOffset(years=20)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    np.random.seed(42)
    n_days = len(date_range)
    
    # Simulate realistic pharmaceutical sales patterns
    trend = np.linspace(1000, 1800, n_days)  # Growing trend over 5 years
    weekly_pattern = 200 * np.sin(2 * np.pi * np.arange(n_days) / 7)  # Weekly seasonality
    monthly_pattern = 100 * np.sin(2 * np.pi * np.arange(n_days) / 30)  # Monthly pattern
    quarterly_pattern = 150 * np.sin(2 * np.pi * np.arange(n_days) / 90)  # Quarterly pattern
    noise = np.random.normal(0, 50, n_days)  # Random noise
    
    # Combine components
    sales = trend + weekly_pattern + monthly_pattern + quarterly_pattern + noise
    sales = np.maximum(sales, 50)  # Ensure minimum sales
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': date_range,
        'sales_amount': sales
    })
    
    return df


def main():
    """Main function to test the standalone Prophet model"""
    
    print("="*60)
    print("STANDALONE PROPHET MODEL TEST")
    print("="*60)
    
    try:
        # CLI arguments
        parser = argparse.ArgumentParser(description='Prophet forecasting for pharmaceutical sales')
        parser.add_argument('--data-path', type=str, default='data', help='Path to CSV file or directory with CSVs')
        parser.add_argument('--date-col', type=str, default=None, help='Date column name in CSV')
        parser.add_argument('--target-col', type=str, default=None, help='Target column name in CSV')
        parser.add_argument('--horizon', type=int, default=None, help='Forecast horizon in periods (days)')
        parser.add_argument('--test-size', type=float, default=None, help='Test size fraction for evaluation')
        parser.add_argument('--generate-fake', action='store_true', help='Generate 5-year daily fake data CSVs in data/ and exit')
        args = parser.parse_args()

        # Handle fake data generation only
        if args.generate_fake:
            print("\n🧪 Generating 5-year daily fake datasets...")
            fake_df = create_sample_data()
            Path('data').mkdir(exist_ok=True)
            fake_df.to_csv('data/sample_sales.csv', index=False)
            fake_cli = fake_df.rename(columns={'date': 'InvoiceDate', 'sales_amount': 'Sales'})
            fake_cli.to_csv('data/my_sales.csv', index=False)
            print("   Saved: data/sample_sales.csv (date, sales_amount)")
            print("   Saved: data/my_sales.csv (InvoiceDate, Sales)")
            return

        # Load or create configuration
        try:
            with open('config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            print("✅ Configuration loaded from config.yaml")
        except FileNotFoundError:
            print("⚠️ config.yaml not found, using default configuration")
            config = {
                'data': {
                    'date_column': 'date',
                    'target_column': 'sales_amount',
                    'forecast_horizon': 30,
                    'test_size': 0.2
                },
                'models': {
                    'prophet': {
                        'daily_seasonality': False,  # Changed to False for daily data
                        'weekly_seasonality': True,
                        'yearly_seasonality': True,
                        'seasonality_mode': 'additive'
                    }
                }
            }
        # Apply CLI overrides
        if args.date_col:
            config['data']['date_column'] = args.date_col
        if args.target_col:
            config['data']['target_column'] = args.target_col
        if args.horizon is not None:
            config['data']['forecast_horizon'] = int(args.horizon)
        if args.test_size is not None:
            config['data']['test_size'] = float(args.test_size)
        
        # Ensure data directory exists to guide users
        Path('data').mkdir(exist_ok=True)
         
        # Load user data if available, otherwise create sample data
        user_df = load_data_from_csv(args.data_path, config['data']['date_column'], config['data']['target_column'])
        if user_df is not None:
            print(f"\n📊 Loaded data from {args.data_path}")
            data = user_df
            # auto-detect columns if configured ones are missing
            try:
                cols = set(map(str, data.columns))
                desired_date = config['data']['date_column']
                desired_target = config['data']['target_column']
                if desired_date not in cols or desired_target not in cols:
                    # try common schemas
                    if {'date', 'sales_amount'}.issubset(cols):
                        config['data']['date_column'] = 'date'
                        config['data']['target_column'] = 'sales_amount'
                    elif {'InvoiceDate', 'Sales'}.issubset(cols):
                        config['data']['date_column'] = 'InvoiceDate'
                        config['data']['target_column'] = 'Sales'
            except Exception:
                pass
            # quick summary
            try:
                print(f"   Rows: {len(data)}")
                print(f"   Date range: {pd.to_datetime(data[config['data']['date_column']]).min().date()} to {pd.to_datetime(data[config['data']['date_column']]).max().date()}")
                print(f"   Target min/max: {pd.to_numeric(data[config['data']['target_column']], errors='coerce').min():.2f} / {pd.to_numeric(data[config['data']['target_column']], errors='coerce').max():.2f}")
            except Exception:
                pass
        else:
            print("\n📊 No CSV found. Creating sample pharmaceutical sales data...")
            data = create_sample_data()
            print(f"   Generated {len(data)} days of sales data")
            print(f"   Date range: {data['date'].min().date()} to {data['date'].max().date()}")
            print(f"   Sales range: ${data['sales_amount'].min():.2f} to ${data['sales_amount'].max():.2f}")
            # Ensure config columns match sample data
            config['data']['date_column'] = 'date'
            config['data']['target_column'] = 'sales_amount'
            # Save fake data to CSVs for convenience
            try:
                Path('data').mkdir(exist_ok=True)
                data.to_csv('data/sample_sales.csv', index=False)
                # Also provide a version matching common column names used in CLI examples
                data_cli = data.rename(columns={'date': 'InvoiceDate' , 'sales_amount': 'Sales'})
                data_cli.to_csv('data/my_sales.csv', index=False)
                print("   Saved fake datasets: data/sample_sales.csv and data/my_sales.csv")
            except Exception as _e:
                logger.warning(f"Could not save fake CSVs: {_e}")
         
        # Initialize model
        print("\n🤖 Initializing Prophet model...")
        prophet_model = StandalonePharmaceuticalProphetModel(config)
         
        # Prepare data
        print("\n📈 Preparing data...")
        train_data, test_data = prophet_model.prepare_data(
            data, 
            config['data']['date_column'],
            config['data']['target_column'],
            config['data']['test_size']
        )
        
        if train_data.empty:
            raise ValueError("Failed to prepare training data")
        
        # Fit model
        print("\n⏳ Training Prophet model... (this may take 30-60 seconds)")
        fit_success = prophet_model.fit_model(train_data)
        
        if not fit_success:
            raise ValueError("Failed to fit Prophet model")
        
        # Evaluate on test data
        print("\n📊 Evaluating model performance...")
        metrics = prophet_model.evaluate_model(test_data)
        
        if metrics:
            print("   Model Performance:")
            print(f"     RMSE: ${metrics['rmse']:.2f}")
            print(f"     MAE: ${metrics['mae']:.2f}")
            print(f"     R² Score: {metrics['r2_score']:.3f}")
            print(f"     MAPE: {metrics['mape']:.2f}%")
            print(f"     Directional Accuracy: {metrics['directional_accuracy']:.1f}%")
         
        # Generate forecast
        print(f"\n🔮 Generating {config['data']['forecast_horizon']}-day forecast...")
        forecast_df = prophet_model.forecast(periods=config['data']['forecast_horizon'])
         
        if not forecast_df.empty:
            future_forecast = forecast_df.tail(config['data']['forecast_horizon'])
            print("   Forecast Summary:")
            print(f"     Average daily sales: ${future_forecast['yhat'].mean():.2f}")
            print(f"     Total forecasted sales: ${future_forecast['yhat'].sum():.2f}")
            print(f"     Forecast range: ${future_forecast['yhat'].min():.2f} - ${future_forecast['yhat'].max():.2f}")
        else:
            raise ValueError("Failed to generate forecast")
         
        # Save model and results
        print("\n💾 Saving results...")
        save_success = prophet_model.save_model("models/standalone_prophet_model.pkl")
        
        if save_success and not forecast_df.empty:
            # Create forecasts directory if it doesn't exist
            Path("forecasts").mkdir(exist_ok=True)
            forecast_df.to_csv("forecasts/prophet_forecast.csv", index=False)
            print("   ✅ Forecast saved to forecasts/prophet_forecast.csv")
         
        # Create visualization
        print("\n📊 Creating forecast visualization...")
        historical_data = pd.DataFrame({
            'ds': pd.to_datetime(data[config['data']['date_column']]),
            'y': pd.to_numeric(data[config['data']['target_column']], errors='coerce')
        })
        
        # Create plots directory if it doesn't exist
        Path("plots").mkdir(exist_ok=True)
        
        prophet_model.plot_forecast(
            historical_data=historical_data,
            save_path="plots/prophet_forecast.png"
        )
        
        print("\n🎉 Standalone Prophet model test completed successfully!")
        print("="*60)
         
    except Exception as e:
        print(f"\n❌ Error in main function: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()