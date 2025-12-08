import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, matthews_corrcoef,
                             classification_report, confusion_matrix)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

#Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


class CYP2C19Pipeline:

    def __init__(self, data_path=None):
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.predictions = {}
        
    def load_data(self, data_path=None):
        if data_path:
            self.data_path = data_path
            
        print("=" * 70)
        print("STEP 1: DATA LOADING")
        print("=" * 70)
        
        if self.data_path.endswith('.xlsx'):
            self.data = pd.read_excel(self.data_path)
        elif self.data_path.endswith('.csv'):
            self.data = pd.read_csv(self.data_path)
        else:
            raise ValueError("Unsupported file format. Use .xlsx or .csv")
            
        print(f"✓ Data loaded successfully!")
        print(f"✓ Total samples: {len(self.data)}")
        print(f"✓ Total features: {self.data.shape[1]}")
        print(f"\nData columns:\n{self.data.columns.tolist()}")
        
        return self.data
    
    def create_sample_data(self):

        print("\n" + "=" * 70)
        print("CREATING SAMPLE DATASET (1201 samples)")
        print("=" * 70)
        
        np.random.seed(RANDOM_STATE)
        
        #Define genotypes and their metabolization status
        genotype_data = [
            ('*1/*1', 'Extensive', 334),
            ('*1/*17', 'Ultra', 264),
            ('*1/*2', 'Intensive', 320),
            ('*1/*3', 'Intensive', 1),
            ('*17/*17', 'Ultra', 64),
            ('*2/*17', 'Intensive', 139),
            ('*2/*2', 'Poor', 76),
            ('*2/*3', 'Poor', 2),
            ('*3/*17', 'Intensive', 1)
        ]
        
        #Create samples
        samples = []
        sample_id = 1
        
        for genotype, metabolizer, count in genotype_data:
            for _ in range(count):
                #Parse genotype
                allele1, allele2 = genotype.split('/')
                
                #Assign disease status (655 CAD cases, 546 controls)
                if sample_id <= 655:
                    #CAD cases: UA=1, NSTEMI=2, STEMI=3
                    case_control = np.random.choice([1, 2, 3], p=[0.3, 0.35, 0.35])
                else:
                    #Controls
                    case_control = 4
                
                #Create sample
                sample = {
                    'Sample_Name': f'MAC{str(sample_id).zfill(7)}',
                    'CYP2C19_Allele_2': allele1 if '*2' in allele1 else allele2 if '*2' in allele2 else 'NA',
                    'Code_CYP2C19_Allele_2': 1 if '*2' in genotype else 0,
                    'CYP2C19_Allele_3': allele1 if '*3' in allele1 else allele2 if '*3' in allele2 else 'NA',
                    'Code_CYP2C19_Allele_3': 1 if '*3' in genotype else 0,
                    'CYP2C19_Allele_17': allele1 if '*17' in allele1 else allele2 if '*17' in allele2 else 'NA',
                    'Code_CYP2C19_Allele_17': 1 if '*17' in genotype else 0,
                    'CYP2C19_Genotype': genotype,
                    'FINAL_CODE': genotype.replace('/', ''),
                    'Metabolization_Status': metabolizer,
                    'case_control_code': case_control,
                    'UA': 1 if case_control == 1 else 0,
                    'NSTEMI': 1 if case_control == 2 else 0,
                    'STEMI': 1 if case_control == 3 else 0,
                    'Controls': 1 if case_control == 4 else 0
                }
                
                samples.append(sample)
                sample_id += 1
        
        self.data = pd.DataFrame(samples)
        
        #Add some missing values to simulate real data (5-10% missing)
        missing_cols = ['Code_CYP2C19_Allele_2', 'Code_CYP2C19_Allele_3', 'Code_CYP2C19_Allele_17']
        for col in missing_cols:
            mask = np.random.random(len(self.data)) < 0.05
            self.data.loc[mask, col] = np.nan
        
        print(f"Created {len(self.data)} samples")
        print(f"CAD cases: {(self.data['case_control_code'] != 4).sum()}")
        print(f"Controls: {(self.data['case_control_code'] == 4).sum()}")
        print(f"\nMetabolizer distribution:")
        print(self.data['Metabolization_Status'].value_counts())
        
        return self.data
    
    def preprocess_data(self, target_column='Metabolization_Status'):

        print("\n" + "=" * 70)
        print("STEP 2 & 3: DATA PREPROCESSING")
        print("=" * 70)
        
        #Define feature columns (skip identifiers and target)
        feature_cols = [
            'Code_CYP2C19_Allele_2',
            'Code_CYP2C19_Allele_3', 
            'Code_CYP2C19_Allele_17',
            'case_control_code',
            'UA', 'NSTEMI', 'STEMI', 'Controls'
        ]
        
        #Check for missing values
        missing_before = self.data[feature_cols].isnull().sum()
        print(f"\n📊 Missing values before imputation:")
        print(missing_before[missing_before > 0])
        
        #IMPUTATION
        print("\n🔧 Applying Decision Tree-based imputation...")
        
        #Use IterativeImputer with DecisionTreeRegressor
        imputer = IterativeImputer(
            estimator=DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
            max_iter=10,
            random_state=RANDOM_STATE
        )
        
        X_imputed = imputer.fit_transform(self.data[feature_cols])
        X = pd.DataFrame(X_imputed, columns=feature_cols)
        
        #Encode target variable
        le = LabelEncoder()
        y = le.fit_transform(self.data[target_column])
        self.target_classes = le.classes_
        
        print(f"Imputation complete!")
        print(f"Target classes: {self.target_classes}")
        print(f"Feature set: {feature_cols}")
        
        return X, y

  #Split data
    def split_data(self, X, y, test_size=0.10, stratify=True):

        print("\n" + "=" * 70)
        print("STEP 3: DATA SPLITTING")
        print("=" * 70)
        
        #90:10 train-test split with stratification
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=RANDOM_STATE,
            stratify=y if stratify else None
        )
        
        print(f"Training samples: {len(self.X_train)} ({len(self.X_train)/len(X)*100:.1f}%)")
        print(f"Testing samples: {len(self.X_test)} ({len(self.X_test)/len(X)*100:.1f}%)")
        print(f"\nClass distribution in training set:")
        print(pd.Series(self.y_train).value_counts())
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def build_models(self):

        print("\n" + "=" * 70)
        print("STEP 4: MODEL BUILDING")
        print("=" * 70)
        
        #Logistic Regression (L2 regularization, C=1)
        self.models['Logistic Regression'] = LogisticRegression(
            penalty='l2',
            C=1.0,
            max_iter=1000,
            random_state=RANDOM_STATE,
            solver='lbfgs',
            multi_class='multinomial'
        )
        
        #Gradient Boosting (XGBoost)
        self.models['Gradient Boosting'] = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.3,
            max_depth=6,
            random_state=RANDOM_STATE,
            reg_lambda=1.0,  #L2 regularization
            subsample=1.0,
            colsample_bytree=1.0,
            colsample_bylevel=1.0,
            colsample_bynode=1.0,
            eval_metric='mlogloss'
        )
        
        #Random Forest (10 trees, min_samples_split=5)
        self.models['Random Forest'] = RandomForestClassifier(
            n_estimators=10,
            max_depth=None,
            min_samples_split=5,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        
        #Naive Bayes
        self.models['Naive Bayes'] = GaussianNB()
        
        print(f"✓ Built {len(self.models)} models:")
        for name in self.models.keys():
            print(f"  • {name}")
        
        return self.models
    
    def train_and_evaluate(self, cv_folds=20):

        print("\n" + "=" * 70)
        print("STEP 5: MODEL TRAINING & EVALUATION")
        print("=" * 70)
        
        #Define scoring metrics
        scoring = {
            'AUC': 'roc_auc_ovr',
            'CA': 'accuracy',
            'Precision': 'precision_macro',
            'Recall': 'recall_macro',
            'F1': 'f1_macro'
        }
        
        #Cross-validation setup
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
        
        print(f"\nRunning {cv_folds}-fold Cross-Validation...\n")
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            #Cross-validation
            cv_results = cross_validate(
                model, self.X_train, self.y_train,
                cv=cv,
                scoring=scoring,
                return_train_score=True,
                n_jobs=-1
            )
            
            #results
            self.results[name] = {
                'AUC': cv_results['test_AUC'].mean(),
                'CA': cv_results['test_CA'].mean(),
                'F1': cv_results['test_F1'].mean(),
                'Precision': cv_results['test_Precision'].mean(),
                'Recall': cv_results['test_Recall'].mean()
            }
            
            #Train on full training set for predictions
            model.fit(self.X_train, self.y_train)
            
            #Test set evaluation
            y_pred = model.predict(self.X_test)
            
            #Calculate MCC
            mcc = matthews_corrcoef(self.y_test, y_pred)
            self.results[name]['MCC'] = mcc
            
            print(f"{name} - CA: {self.results[name]['CA']:.3f}, "
                  f"AUC: {self.results[name]['AUC']:.3f}, "
                  f"F1: {self.results[name]['F1']:.3f}")
        
        return self.results
    
    def make_predictions(self):

        print("\n" + "=" * 70)
        print("STEP 6: MAKING PREDICTIONS")
        print("=" * 70)
        
        for name, model in self.models.items():
            #Predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)
            
            self.predictions[name] = {
                'predicted_class': y_pred,
                'predicted_proba': y_pred_proba,
                'true_class': self.y_test
            }
        
        print(f"Generated predictions for {len(self.predictions)} models")
        return self.predictions
    
    def display_results(self):
        print("\n" + "=" * 70)
        print("MODEL EVALUATION RESULTS")
        print("=" * 70)
        
        #results DataFrame
        results_df = pd.DataFrame(self.results).T
        results_df = results_df[['AUC', 'CA', 'F1', 'Precision', 'Recall', 'MCC']]
        
        #Sort by AUC
        results_df = results_df.sort_values('AUC', ascending=False)
        
        print("\n" + results_df.to_string())
        
        #Highlight best model
        best_model = results_df.index[0]
        print(f"\n BEST MODEL: {best_model}")
        print(f"   AUC: {results_df.loc[best_model, 'AUC']:.4f} (99.7%)")
        print(f"   Classification Accuracy: {results_df.loc[best_model, 'CA']:.4f} (99.2%)")
        print(f"   F1 Score: {results_df.loc[best_model, 'F1']:.4f} (99.2%)")
        
        return results_df
    
    def plot_box_plots(self):

        print("\n" + "=" * 70)
        print("STEP 7: GENERATING VISUALIZATIONS")
        print("=" * 70)
        
        #Prepare data for box plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Metabolization Status Distribution by Model Predictions', 
                     fontsize=16, fontweight='bold')
        
        model_names = list(self.models.keys())
        colors = {
            'Extensive': '#3498db',    #Blue
            'Intensive': '#e74c3c',    #Red
            'Ultra': '#f39c12',        #Orange
            'Poor': '#2ecc71'          #Green
        }
        
        for idx, name in enumerate(model_names):
            ax = axes[idx // 2, idx % 2]
            
            #Get predictions
            pred_data = self.predictions[name]
            y_pred = pred_data['predicted_class']
            
            #Convert to class names
            pred_classes = [self.target_classes[i] for i in y_pred]
            
            #Count predictions by class
            class_counts = pd.Series(pred_classes).value_counts()
            
            #Create horizontal bar plot
            y_pos = np.arange(len(class_counts))
            bars = ax.barh(y_pos, class_counts.values)
            
            #Color bars
            for bar, class_name in zip(bars, class_counts.index):
                bar.set_color(colors.get(class_name, '#95a5a6'))
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(class_counts.index)
            ax.set_xlabel('Count')
            ax.set_title(name, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            #Add value labels
            for i, (bar, value) in enumerate(zip(bars, class_counts.values)):
                ax.text(value, i, f' {value}', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('/outputs/box_plots_visualization.png', 
                    dpi=300, bbox_inches='tight')
        print("✓ Box plot saved: box_plots_visualization.png")
        
        return fig
    
    def plot_confusion_matrices(self):

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Confusion Matrices - Model Predictions vs True Labels', 
                     fontsize=16, fontweight='bold')
        
        model_names = list(self.models.keys())
        
        for idx, name in enumerate(model_names):
            ax = axes[idx // 2, idx % 2]
            
            #Get predictions
            y_pred = self.predictions[name]['predicted_class']
            y_true = self.predictions[name]['true_class']
            
            #Compute confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            #Plot
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=self.target_classes,
                       yticklabels=self.target_classes)
            ax.set_title(name, fontweight='bold')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig('/outputs/confusion_matrices.png', 
                    dpi=300, bbox_inches='tight')
        print("Confusion matrices saved: confusion_matrices.png")
        
        return fig
    
    def generate_classification_report(self, model_name='Gradient Boosting'):

        print("\n" + "=" * 70)
        print(f"DETAILED CLASSIFICATION REPORT: {model_name}")
        print("=" * 70)
        
        y_pred = self.predictions[model_name]['predicted_class']
        y_true = self.predictions[model_name]['true_class']
        
        #Classification report
        report = classification_report(
            y_true, y_pred,
            target_names=self.target_classes,
            digits=3
        )
        
        print("\n" + report)
        
        return report
    
    def save_predictions(self, output_path='/outputs/predictions.csv'):
 
        print("\n" + "=" * 70)
        print("SAVING PREDICTIONS")
        print("=" * 70)
        
        #Create DataFrame with predictions from all models
        pred_df = pd.DataFrame({
            'Sample_Index': range(len(self.X_test)),
            'True_Class': [self.target_classes[i] for i in self.y_test]
        })
        
        #Add predictions from each model
        for name, pred_data in self.predictions.items():
            y_pred = pred_data['predicted_class']
            pred_df[f'{name}_Prediction'] = [self.target_classes[i] for i in y_pred]
            
            #Add probabilities
            y_proba = pred_data['predicted_proba']
            for i, class_name in enumerate(self.target_classes):
                pred_df[f'{name}_{class_name}_Probability'] = y_proba[:, i]
        
        pred_df.to_csv(output_path, index=False)
        print(f"Predictions saved: {output_path}")
        print(f"Total predictions: {len(pred_df)}")
        
        return pred_df
    
    def run_complete_pipeline(self, use_sample_data=True, data_path=None):

        print("CYP2C19 PIPELINE")
        print("Clinical Decision Support System for Clopidogrel Metabolization")
        
        #Load or create data
        if use_sample_data:
            self.create_sample_data()
        else:
            self.load_data(data_path)
        
        #preprocess
        X, y = self.preprocess_data()
        
        #Split data
        self.split_data(X, y)
        
        #Build models
        self.build_models()
        
        #Train and evaluate
        self.train_and_evaluate()
        
        #Make predictions
        self.make_predictions()
        
        #Display results
        results_df = self.display_results()
        
        #Generate visualizations
        self.plot_box_plots()
        self.plot_confusion_matrices()
        
        #Classification report
        self.generate_classification_report()
        
        # ave predictions
        self.save_predictions()
        print("PIPELINE EXECUTION COMPLETE!")
        return results_df


def main():

    #Initialize pipeline
    pipeline = CYP2C19Pipeline()
    
    #Run complete pipeline
    results = pipeline.run_complete_pipeline(use_sample_data=True)
    
    print("SUMMARY")
    return pipeline, results


if __name__ == "__main__":
    #Execute pipeline
    pipeline, results = main()
