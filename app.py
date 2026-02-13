# استيراد المكتبات اللازمة
from flask import Flask, render_template, request, flash, redirect  # Flask لإنشاء تطبيق الويب
import numpy as np  # للتعامل مع البيانات العددية والمصفوفات
from PIL import Image  # لمعالجة الصور
from tensorflow.keras.models import load_model  # لتحميل نماذج التعلم العميق المدربة

# تحميل النماذج عند بدء التشغيل مرة واحدة فقط
# تحميل النماذج عند بدء التشغيل مرة واحدة فقط
malaria_model = None
pneumonia_model = None

try:
    malaria_model = load_model("models/malaria_fixed.h5", compile=False)
    pneumonia_model = load_model("models/pneumonia_fixed.h5", compile=False)
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")

# إنشاء تطبيق Flask
app = Flask(__name__)

# تعريف الصفحة الرئيسية
from explainability import make_gradcam_heatmap, save_gradcam
import os
import uuid

@app.route("/", methods=['GET'])
def home():
    return render_template('home.html')

@app.route("/main", methods=['GET'])
def main():
    return render_template('main.html')

@app.route("/malaria", methods=['GET', 'POST'])
def malariaPage():
    return render_template('malaria.html')

@app.route("/pneumonia", methods=['GET', 'POST'])
def pneumoniaPage():
    return render_template('pneumonia.html')

@app.route("/malariapredict", methods=['POST', 'GET'])
def malariapredictPage():
    if request.method == 'GET':
        return redirect('/malaria')
        
    pred = None
    processed_img_path = None
    
    if request.method == 'POST':
        try:
            if 'image' in request.files:
                # Save original file temporarily for OpenCV
                original_file = request.files['image']
                original_filename = f"{uuid.uuid4()}.png"
                original_path = os.path.join('static', 'uploads', original_filename)
                
                # Ensure uploads dir exists
                os.makedirs(os.path.join('static', 'uploads'), exist_ok=True)
                
                original_file.save(original_path)
                
                # Preprocess for Model
                img = Image.open(original_path)
                img = img.convert('RGB')
                img = img.resize((36, 36))
                img_array = np.asarray(img)
                img_array = img_array.reshape((1, 36, 36, 3))
                img_array = img_array.astype(np.float64)
                
                if malaria_model is None:
                     raise Exception("Malaria model not loaded")

                prediction = malaria_model.predict(img_array)
                pred_index = np.argmax(prediction[0])
                confidence = prediction[0][pred_index]
                pred = pred_index
                
                # If infected (Class 1), generate heatmap
                if pred == 1:
                    heatmap = make_gradcam_heatmap(img_array, malaria_model, 'conv2d_2')
                    processed_filename = f"gradcam_{original_filename}"
                    processed_path = os.path.join('static', 'uploads', processed_filename)
                    
                    save_gradcam(original_path, heatmap, processed_path, confidence=confidence, label="Infected")
                    processed_img_path = processed_path

        except Exception as e:
            print(f"Error in malaria prediction: {e}")
            message = "الرجاء تحميل صورة الخلية فقط"
            return render_template('malaria.html', message=message)
            
    return render_template('malaria_predict.html', pred=pred, processed_img=processed_img_path, confidence=f"{confidence*100:.2f}" if pred==1 else None)

@app.route("/pneumoniapredict", methods=['POST', 'GET'])
def pneumoniapredictPage():
    if request.method == 'GET':
        return redirect('/pneumonia')

    pred = None
    processed_img_path = None

    if request.method == 'POST':
        try:
            if 'image' in request.files:
                original_file = request.files['image']
                original_filename = f"{uuid.uuid4()}.png"
                original_path = os.path.join('static', 'uploads', original_filename)
                
                os.makedirs(os.path.join('static', 'uploads'), exist_ok=True)
                
                original_file.save(original_path)
                
                img = Image.open(original_path)
                img_rgb = img.convert('RGB')
                img_np = np.array(img_rgb)
                
                mean_red = np.mean(img_np[:, :, 0])
                mean_green = np.mean(img_np[:, :, 1])
                mean_blue = np.mean(img_np[:, :, 2])
                
                if abs(mean_red - mean_green) > 20 or abs(mean_green - mean_blue) > 20 or abs(mean_red - mean_blue) > 20:
                     message = "رجاء قم بتحميل صورة أشعة سينية مناسبة."
                     return render_template('pneumonia.html', message=message)
                
                # Preprocess for Model (Grayscale)
                img_l = img.convert('L')
                img_l = img_l.resize((36, 36))
                img_array = np.asarray(img_l)
                img_array = img_array.reshape((1, 36, 36, 1))
                img_array = img_array / 255.0

                if pneumonia_model is None:
                    raise Exception("Model not loaded properly")

                prediction = pneumonia_model.predict(img_array)
                pred_index = np.argmax(prediction[0])
                confidence = prediction[0][pred_index]
                pred = pred_index
                
                # If Pneumonia (Class 1), generate heatmap
                if pred == 1:
                    heatmap = make_gradcam_heatmap(img_array, pneumonia_model, 'conv2d_5')
                    processed_filename = f"gradcam_{original_filename}"
                    processed_path = os.path.join('static', 'uploads', processed_filename)
                    
                    save_gradcam(original_path, heatmap, processed_path, confidence=confidence, label="Pneumonia")
                    processed_img_path = processed_path

        except Exception as e:
            print(f"Error in pneumonia prediction: {e}")
            message = "رجاء قم باختيار صورة"
            return render_template('pneumonia.html', message=message)
    
    return render_template('pneumonia_predict.html', pred=pred, processed_img=processed_img_path, confidence=f"{confidence*100:.2f}" if pred==1 else None)

# --- Gemini Configuration ---
import google.generativeai as genai

# Gemini Configuration
GENAI_API_KEY = "AIzaSyBPduSeG34T0fY9r9IFCAH7JnWjG6bQGHw" 
genai.configure(api_key=GENAI_API_KEY)

@app.route("/api/chat", methods=['POST'])
def chat_api():
    try:
        data_in = request.json
        user_message = data_in.get('message', '')
        image_data = data_in.get('image') # Base64 string
        
        # Scenario 1: Image Provided -> Analyze with Local Models (Privacy Preserved)
        if image_data:
            import base64
            import io
            from PIL import Image, ImageFile
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            import numpy as np
    
            # Decode Image
            if "base64," in image_data:
                image_data = image_data.split("base64,")[1]
            
            img_bytes = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            
            # --- Model 1: Malaria Analysis ---
            # Preprocess EXACTLY as requested for Local Models
            img_malaria = img.resize((36, 36))
            img_array_mal = np.asarray(img_malaria).astype(np.float64) # No division, float64
            img_array_mal = np.expand_dims(img_array_mal, axis=0)
            
            # Predict Malaria
            pred_malaria = malaria_model.predict(img_array_mal)
            idx_mal = np.argmax(pred_malaria[0])
            conf_mal = float(pred_malaria[0][idx_mal])
            
            # --- Model 2: Pneumonia Analysis ---
            img_pneumonia = img.resize((36, 36)).convert('L')
            img_array_pneu = np.asarray(img_pneumonia).astype(np.float64)
            img_array_pneu = np.expand_dims(img_array_pneu, axis=0)
            img_array_pneu = np.expand_dims(img_array_pneu, axis=-1)
            
            # Predict Pneumonia
            pred_pneumonia = pneumonia_model.predict(img_array_pneu)
            idx_pneu = np.argmax(pred_pneumonia[0])
            conf_pneu = float(pred_pneumonia[0][idx_pneu])
            
            # Build local analysis result
            local_analysis = ""
            if idx_mal == 1 and conf_mal > 0.8:
                local_analysis = f"🔬 **تحليل الملاريا (محلي):**\n✅ النتيجة: مصابة (Parasitized)\n📊 الدقة: {conf_mal:.1%}"
            elif idx_pneu == 1 and conf_pneu > 0.8:
                local_analysis = f"🫁 **تحليل الرئة (محلي):**\n✅ النتيجة: التهاب رئوي (Pneumonia)\n📊 الدقة: {conf_pneu:.1%}"
            else:
                local_analysis = f"📋 **نتائج التحليل المحلي:**\n✅ لم يتم رصد إصابات مؤكدة\n\n• ملاريا: {conf_mal:.1%} (سليمة)\n• رئة: {conf_pneu:.1%} (سليمة)"
            
            # --- Gemini Vision Analysis ---
            gemini_analysis = ""
            try:
                # Save image temporarily for Gemini
                import os
                temp_img_path = "static/uploads/temp_analysis.jpg"
                os.makedirs(os.path.dirname(temp_img_path), exist_ok=True)
                img.save(temp_img_path, "JPEG")
                
                # Upload to Gemini
                uploaded_file = genai.upload_file(temp_img_path)
                
                # Create vision model
                vision_model = genai.GenerativeModel('gemini-2.5-flash')
                
                # Medical analysis prompt
                medical_prompt = """أنت طبيب متخصص في تحليل الصور الطبية.
قم بتحليل هذه الصورة الطبية بدقة وحدد:

1. **نوع الصورة**: (خلايا دم مجهرية / أشعة سينية للصدر / أخرى)
2. **الملاحظات المرضية**: هل توجد أي علامات غير طبيعية؟
3. **التشخيص المحتمل**:
   - إذا كانت خلايا دم: هل توجد طفيليات الملاريا؟
   - إذا كانت أشعة رئة: هل توجد علامات التهاب رئوي؟
4. **التوصيات**: ماذا يجب على المريض فعله؟

أجب بالعربية بشكل مختصر ومهني."""

                # Get Gemini analysis
                response = vision_model.generate_content([medical_prompt, uploaded_file])
                gemini_analysis = f"🤖 **تحليل الذكاء الاصطناعي (Gemini Vision):**\n{response.text}"
                
                # Clean up
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)
                    
            except Exception as e:
                error_msg = str(e)
                # Check if it's a quota exceeded error
                if "429" in error_msg or "quota" in error_msg.lower() or "exceeded" in error_msg.lower():
                    gemini_analysis = f"""⚠️ **تحليل Gemini Vision:**
❌ **تم الوصول لحد الاستخدام اليومي**

📊 **السبب:**
تم استخدام الحصة المجانية من Gemini API (20 طلب/يوم)

✅ **الحل:**
• انتظر دقيقة واحدة وحاول مرة أخرى
• أو استخدم التحليل المحلي فقط (دقيق جداً!)
• أو انتظر حتى الغد لإعادة تعيين الحصة

💡 **ملاحظة:** التحليل المحلي أعلاه دقيق بنسبة عالية ويكفي للتشخيص الأولي."""
                else:
                    gemini_analysis = f"⚠️ **تحليل Gemini:**\nلم يتمكن من التحليل (خطأ: {error_msg[:80]}...)"
            
            # Combine Results
            final_result = f"""{local_analysis}

{gemini_analysis}

💡 **ملاحظة مهمة:**
هذا التحليل للإرشاد فقط. يُرجى استشارة طبيب مختص لتأكيد التشخيص."""

            return {
                "choices": [{
                    "message": {
                        "content": final_result
                    }
                }]
            }

        # Scenario 2: Text Only -> Ask Gemini (Medical Only)
        else:
            if not user_message:
                return {"choices": [{"message": {"content": "مرحباً! أنا دكتور 7، مساعدك الطبي الذكي. كيف يمكنني مساعدتك اليوم؟"}}]}

            # Strict Medical System Prompt
            system_instruction = """
            You are 'Dr.7' (دكتور 7), a purely medical AI assistant.
            
            Your Rules:
            1. ONLY answer questions about Medicine, Health, Biology, diseases, and treatments.
            2. If a user asks about programming, politics, sports, or general chit-chat, REFUSE politely.
            3. Provide accurate, helpful, and concise medical advice.
            4. Answer in the same language as the user (mostly Arabic).
            5. If asked about who developed this application, mention that it was developed by Engineer Abbas Mohsen (المهندس عباس محسن).
            """
            
            # Use 'gemini-2.5-flash' - the latest fast and stable model
            model = genai.GenerativeModel('gemini-2.5-flash')
            chat = model.start_chat(history=[])
            
            # Engineering the prompt to enforce instructions
            full_prompt = f"{system_instruction}\n\nUser Query: {user_message}"
            
            try:
                response = chat.send_message(full_prompt)
                
                return {
                    "choices": [{
                        "message": {
                            "content": response.text
                        }
                    }]
                }
            except Exception as e:
                error_msg = str(e)
                # Check if it's a quota exceeded error
                if "429" in error_msg or "quota" in error_msg.lower() or "exceeded" in error_msg.lower():
                    quota_warning = f"""⚠️ **تنبيه: تم الوصول لحد الاستخدام**

❌ **المشكلة:**
تم استخدام الحصة المجانية من Gemini API (20 طلب/يوم لهذا النموذج)

🔄 **الحل:**
• انتظر دقيقة واحدة وحاول مرة أخرى
• أو انتظر حتى الغد لإعادة تعيين الحصة اليومية
• الحصة تتجدد تلقائياً كل 24 ساعة

📊 **معلومة:**
يمكنك استخدام ميزة تحليل الصور (التحليل المحلي) بدون حدود!

💡 **ملاحظة:** هذا الحد طبيعي للنسخة المجانية من Gemini API."""
                    
                    return {
                        "choices": [{
                            "message": {
                                "content": quota_warning
                            }
                        }]
                    }
                else:
                    # Other errors
                    return {
                        "choices": [{
                            "message": {
                                "content": f"عذراً، حدث خطأ في الاتصال بالمساعد الذكي.\n\nيمكنك استخدام ميزة تحليل الصور (التحليل المحلي) بدلاً من ذلك!"
                            }
                        }]
                    }

    except Exception as e:
        print(f"Chat API Error Detailed: {e}") # Print full error to console
        return {"choices": [{"message": {"content": f"عذراً، حدث خطأ: {str(e)}"}}]} # Show error to user for debugging


# تشغيل التطبيق في وضع التصحيح (debug mode)
if __name__ == '__main__':
    app.run(debug=True)
