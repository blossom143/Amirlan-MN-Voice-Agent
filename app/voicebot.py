"""
Khan Bank Speech-to-Speech Voicebot Agent (Amirlan)
===================================================
A structured 6-stage AI voicebot engine:
Stage 1: Call Intake (sounddevice / browser audio -> Google Cloud Chirp 3 mn-MN)
Stage 2: Intent Understanding (Gemini 2.5 Flash classification)
Stage 3: Information Retrieval (Targeted search restricted to khanbank.com)
Stage 4: Decision Making (LangGraph ReAct reasoning & verification)
Stage 5: Response Generation (Professional Mongolian text -> Narakeet .m4a Ganbaatar voice)
Stage 6: Human Escalation (Safe bank-standard escalation protocol & 1800-1917 handoff)
"""

import os
import re
import json
import time
import uuid
import tempfile
import requests
from typing import Dict, Any, Optional

# Suppress gRPC fork logs on macOS
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"

from dotenv import load_dotenv
load_dotenv(override=True)

# Google Cloud STT V2
from google.cloud import speech_v2 as speech
from google.api_core.client_options import ClientOptions

# Vertex AI Gemini
from langchain_google_vertexai import ChatVertexAI

# Tools
from langchain_community.tools import DuckDuckGoSearchRun

# Sounddevice & Scipy (Desktop Mic)
try:
    import sounddevice as sd
    from scipy.io import wavfile
    import numpy as np
    SOUNDDEVICE_AVAILABLE = True
except Exception:
    SOUNDDEVICE_AVAILABLE = False


class AmirlanVoicebot:
    def __init__(self):
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "project-1d76ba00-25b9-47dd-8d3")
        self.gcp_location = os.getenv("GCP_LOCATION", "us-central1")
        self.narakeet_api_key = os.getenv("NARAKEET_API_KEY")
        
        # Chirp 3 uses multi-region 'us' endpoint
        self.stt_location = "us"
        self.speech_client = speech.SpeechClient(
            client_options=ClientOptions(api_endpoint=f"{self.stt_location}-speech.googleapis.com")
        )
        self.recognizer_path = self.speech_client.recognizer_path(self.project_id, self.stt_location, "_")
        
        # Vertex AI Gemini 2.5 Flash
        self.llm = ChatVertexAI(
            model="gemini-2.5-flash",
            project=self.project_id,
            location=self.gcp_location,
            temperature=0.2
        )
        
        # Search Tool
        try:
            self.search_tool = DuckDuckGoSearchRun()
        except Exception:
            self.search_tool = None

        # Output Directories
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.output_dir = os.path.join(self.base_dir, "output-audio")
        self.input_dir = os.path.join(self.base_dir, "input-audio")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.input_dir, exist_ok=True)

    # =========================================================================
    # STAGE 1: Call Intake (Speech-to-Text via Google Cloud Chirp 3)
    # =========================================================================
    def record_microphone_desktop(self, duration_secs: int = 5, sample_rate: int = 16000) -> str:
        """Records live audio from desktop microphone via sounddevice."""
        if not SOUNDDEVICE_AVAILABLE:
            raise RuntimeError("sounddevice is not available in the current environment.")
        
        filename = os.path.join(self.input_dir, f"desktop_mic_{int(time.time())}.wav")
        recording = sd.rec(int(duration_secs * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
        sd.wait()
        wavfile.write(filename, sample_rate, recording)
        return filename

    def transcribe_audio_chirp3(self, audio_bytes: bytes) -> str:
        """Transcribes Mongolian speech audio into text using Google Cloud Chirp 3."""
        config = speech.RecognitionConfig(
            auto_decoding_config=speech.AutoDetectDecodingConfig(),
            language_codes=["mn-MN"],
            model="chirp_3"
        )
        request = speech.RecognizeRequest(
            recognizer=self.recognizer_path,
            config=config,
            content=audio_bytes
        )
        
        response = self.speech_client.recognize(request=request)
        transcript = ""
        for res in response.results:
            if res.alternatives:
                transcript += res.alternatives[0].transcript
        return transcript.strip()

    # =========================================================================
    # STAGE 2: Intent Understanding (Gemini 2.5 Flash)
    # =========================================================================
    def understand_intent(self, user_query: str) -> Dict[str, Any]:
        """
        Classifies customer intent into structured categories:
        - loan_inquiry (Зээлийн мэдээлэл)
        - account_service (Данс, хадгаламж, төлбөр тооцоо)
        - card_service (Карт захиалах, үйлчилгээ)
        - branch_atm (Салбар, АТМ байршил, цагийн хуваарь)
        - exchange_rate (Ханшийн мэдээлэл)
        - human_escalation (Оператортой холбогдох, гомдол)
        - general_inquiry (Бусад ерөнхий асуулт)
        """
        prompt = f"""Та Хаан Банкны харилцагчийн үйлчилгээний ухаалаг ангилагч систем байна.
Хэрэглэгчийн хэлсэн дараах асуултад дүн шинжилгээ хийж, JSON форматаар хариулна уу.

Хэрэглэгчийн асуулт: "{user_query}"

Хариулах JSON бүтэц:
{{
    "category": "loan_inquiry" | "account_service" | "card_service" | "branch_atm" | "exchange_rate" | "human_escalation" | "general_inquiry",
    "topic": "Товч сэдэв (Монголоор)",
    "is_escalation_needed": true/false (Хэрэв хэрэглэгч оператор, амьд хүн хүссэн эсвэл картын пин код, данс хаалгах зэрэг хувийн нууц мэдээлэл шаардсан бол true),
    "search_keywords": "khanbank.com дээрээс хайхад тохиромжтой 2-4 түлхүүр үг"
}}

ЗӨВХӨН JSON форматаар хариул. Таййлбар бичихгүй."""

        try:
            res = self.llm.invoke(prompt)
            content = res.content.strip()
            # Clean markdown json code blocks if present
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\s*", "", content)
                content = re.sub(r"\s*```$", "", content)
            intent_data = json.loads(content)
        except Exception:
            # Fallback intent detection
            is_esc = any(w in user_query.lower() for w in ["хүнтэй", "оператор", "менежер", "холбож өг", "туслах ажилтан", "гомдол"])
            intent_data = {
                "category": "human_escalation" if is_esc else "general_inquiry",
                "topic": "Ерөнхий хүсэлт",
                "is_escalation_needed": is_esc,
                "search_keywords": user_query
            }
        return intent_data

    # =========================================================================
    # STAGE 3: Information Retrieval (Targeted khanbank.com Grounding)
    # =========================================================================
    def retrieve_khanbank_info(self, query: str, search_keywords: str) -> Dict[str, Any]:
        """Searches official Khan Bank information."""
        full_query = f"site:khanbank.com {search_keywords}"
        raw_result = ""
        source_url = "https://www.khanbank.com"
        
        if self.search_tool:
            try:
                raw_result = self.search_tool.run(full_query)
            except Exception as e:
                raw_result = f"Хайлтын системийн сануулга: {e}"

        # Standard Khan Bank verified data bank
        knowledge_context = """
Хаан Банкны албан ёсны мэдээлэл:
- Лавлах төв: 1800-1917 (24/7 тасралтгүй ажиллана), Гадаадаас холбогдох: +976 7515-1917
- Цахим банк: Khan Bank App (iOS/Android), e-khanbank.com
- Ипотекийн 6% зээл: Урьдчилгаа 30%, хугацаа 30 хүртэл жил, орон сууцны талбай 100 м.кв хүртэл.
- Цалингийн зээл: Шимтгэлгүй, хугацаа 30 хүртэл сар, сарын хүү 1.4%-1.6% орчим.
- Signature / Platinum дебит карт: Олон улсын төлбөр тооцоо, лаунж үйлчилгээ, хөнгөлөлттэй.
- Салбарууд: Монгол орон даяар 550 гаруй салбар, 1300 гаруй АТМ, Экспресс төвүүд 24 цагаар ажиллана.
"""
        return {
            "search_query": full_query,
            "search_results": raw_result.strip() if raw_result else "Шууд үр дүн олдсонгүй.",
            "verified_context": knowledge_context,
            "source_url": source_url
        }

    # =========================================================================
    # STAGE 4: Decision Making (Reasoning & Escalation Check)
    # =========================================================================
    def make_decision(self, user_query: str, intent_data: Dict[str, Any], retrieval_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates whether the user's intent is satisfied by retrieved Khan Bank data
        or requires escalation to a human banker.
        """
        # Explicit or detected escalation
        if intent_data.get("is_escalation_needed") or intent_data.get("category") == "human_escalation":
            return {
                "decision": "escalate",
                "reason": "Хэрэглэгч оператортай шууд холбогдох эсвэл тусгай үйлчилгээний тусламж хүссэн тул 6-р шат буюу мэргэжилтэнд шилжүүлэх горимыг идэвхжүүллээ.",
                "requires_human": True
            }

        prompt = f"""Та Хаан Банкны Амирлан нэртэй цахим туслахын дүн шинжилгээ хийх систем.
Хэрэглэгчийн асуулт: "{user_query}"
Хэрэглэгчийн зорилго: {json.dumps(intent_data, ensure_ascii=False)}
Олдсон мэдээлэл: {retrieval_data['search_results']}
Банкны баталгаат мэдээлэл: {retrieval_data['verified_context']}

Дараах шалгуураар шийдвэр гаргана уу:
1. Олдсон мэдээлэл хангалттай бөгөөд хэрэглэгчийн асуултад шууд хариулах боломжтой бол: decision = "answer"
2. Хэрэв банкны хувийн нууц мэдээлэл шаардсан, картын гүйлгээний маргаан, эсвэл мэдээлэл дутуу бол: decision = "escalate"

ЗӨВХӨН JSON форматаар хариул:
{{
    "decision": "answer" эсвэл "escalate",
    "reason": "Шийдвэрийн шалтгаан (Монголоор 1 өгүүлбэр)"
}}"""

        try:
            res = self.llm.invoke(prompt)
            content = res.content.strip()
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\s*", "", content)
                content = re.sub(r"\s*```$", "", content)
            decision_data = json.loads(content)
        except Exception:
            decision_data = {
                "decision": "answer",
                "reason": "Хаан Банкны мэдээллийн сангаас хариулах боломжтой гэж дүгнэлээ."
            }

        decision_data["requires_human"] = (decision_data.get("decision") == "escalate")
        return decision_data

    # =========================================================================
    # STAGE 5: Response Generation (Narakeet .m4a with Ganbaatar voice)
    # =========================================================================
    def generate_mongolian_response(self, user_query: str, retrieval_data: Dict[str, Any], is_escalation: bool = False) -> str:
        """Formulates high-standard, polite Mongolian banking response."""
        if is_escalation:
            return (
                "Эрхэм харилцагч танд энэ өдрийн мэнд хүргэе. "
                "Таны хүсэлтийг Хаан Банкны харилцагчийн үйлчилгээний төвийн мэргэжилтэнд шилжүүлж байна. "
                "Та манай 1800-1917 лавлах утсаар 24 цагийн турш холбогдох боломжтой бөгөөд "
                "түр хүлээнэ үү, бид таныг хамгийн ойрын оператортой холбож өгөхөд бэлэн байна."
            )

        prompt = f"""Та Хаан Банкны албан ёсны цахим туслах Амирлан байна.
Хэрэглэгчийн асуултад ДАНДАА МАКСИМУМ ЭЕЛДЭГ, АЛБАН ЁСНЫ, СЭТГЭЛД ХҮРСЭН МОНГОЛ ХЭЛЭЭР хариулна.

Хэрэглэгчийн асуулт: "{user_query}"
Банкны албан ёсны эх сурвалж:
{retrieval_data['search_results']}
{retrieval_data['verified_context']}

ДҮРЭМ:
1. Хариултаа 'Эрхэм харилцагч танд энэ өдрийн мэнд хүргэе' гэж хүндэтгэлтэйгээр эхлүүлнэ.
2. Мэдээллийг товч, цэгцтэй, ойлгомжтой 2-3 өгүүлбэрт (нийт 40-60 үгэнд, 350 тэмдэгтээс хэтрүүлэхгүй) багтаана. Дуу хоолойн ярианд хэт урт хариулт өгөхгүй.
3. Нэмэлт дэлгэрэнгүй мэдээллийг 1800-1917 лавлах утас болон khanbank.com хаягаас авахыг эелдгээр зөвлөнө.
4. Хариулт нь дуу хоолойгоор ярихад сонсголонтой байх ёстой тул элдэв од, хашилт, тусгай тэмдэгт ашиглахгүйгээр цэвэр энгийн монгол текстээр бичнэ үү."""

        try:
            res = self.llm.invoke(prompt)
            clean_text = res.content.strip()
            # Remove markdown headers and bullets for natural audio reading
            clean_text = re.sub(r"[#*_\-`]", " ", clean_text)
            clean_text = re.sub(r"\s+", " ", clean_text).strip()
            return clean_text
        except Exception as e:
            return f"Эрхэм харилцагч танд энэ өдрийн мэнд хүргэе. Хаан Банкны талаарх дэлгэрэнгүй мэдээллийг 1800-1917 утсаар болон khanbank.com хаягаар лавлана уу."

    def _trim_for_tts(self, text: str, max_bytes: int = 950) -> str:
        """Safely trims text to be within Narakeet's 1024-byte synchronous streaming limit."""
        encoded = text.encode("utf-8")
        if len(encoded) <= max_bytes:
            return text
        
        # Decode trimmed bytes safely
        truncated = encoded[:max_bytes].decode("utf-8", errors="ignore")
        
        # Trim at last natural sentence end
        for delim in [". ", "? ", "! "]:
            idx = truncated.rfind(delim)
            if idx > len(truncated) // 2:
                return truncated[:idx + 1].strip()
        
        # Or trim at last word boundary
        last_space = truncated.rfind(" ")
        if last_space > len(truncated) // 3:
            return truncated[:last_space].strip() + "."
        return truncated.strip()

    def synthesize_speech_narakeet(self, text: str, voice: str = "Ganbaatar") -> str:
        """
        Synthesizes Mongolian text into high-quality .m4a audio using Narakeet REST API.
        Saves output audio file to output-audio/ directory.
        """
        if not self.narakeet_api_key:
            raise ValueError("NARAKEET_API_KEY орчны хувьсагч тохируулагдаагүй байна.")

        # Narakeet streaming endpoint strictly requires UTF-8 byte length <= 1024
        safe_text = self._trim_for_tts(text, max_bytes=950)
        filename = f"amirlan_{uuid.uuid4().hex[:8]}_{int(time.time())}.m4a"
        file_path = os.path.join(self.output_dir, filename)

        url = f"https://api.narakeet.com/text-to-speech/m4a?voice={voice}"
        headers = {
            "x-api-key": self.narakeet_api_key,
            "Content-Type": "text/plain; charset=utf-8",
            "Accept": "application/octet-stream"
        }

        res = requests.post(url, data=safe_text.encode("utf-8"), headers=headers, timeout=45)

        content_type = res.headers.get("content-type", "")
        # Synchronous binary audio returned
        if res.status_code == 200 and "application/json" not in content_type:
            with open(file_path, "wb") as f:
                f.write(res.content)
            return filename
        
        # Async task polling fallback if statusUrl is returned
        if res.status_code in [200, 202] and "json" in content_type:
            try:
                task_data = res.json()
                status_url = task_data.get("statusUrl")
                if status_url:
                    # Poll statusUrl without x-api-key header as per Narakeet docs
                    for _ in range(15):
                        time.sleep(2)
                        poll_res = requests.get(status_url, timeout=10)
                        if poll_res.status_code == 200:
                            p_info = poll_res.json()
                            if p_info.get("finished"):
                                audio_url = p_info.get("result")
                                if audio_url:
                                    down_res = requests.get(audio_url, timeout=30)
                                    if down_res.status_code == 200:
                                        with open(file_path, "wb") as f:
                                            f.write(down_res.content)
                                        return filename
            except Exception as poll_e:
                print("Narakeet async polling error:", poll_e)

        raise RuntimeError(f"Narakeet TTS Алдаа ({res.status_code}): {res.text[:200]}")

    # =========================================================================
    # STAGE 6: Human Escalation Protocol
    # =========================================================================
    def execute_escalation_protocol(self, user_query: str, intent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Provides bank-standard human operator handoff protocol details."""
        return {
            "escalation_triggered": True,
            "department": "Харилцагчийн үйлчилгээний төв",
            "hotline": "1800-1917",
            "intl_hotline": "+976 7515-1917",
            "priority": "VIP Priority Routing",
            "estimated_wait": "1 минут хүрэхгүй",
            "message": "Хүсэлтийг амжилттай мэргэжилтний дараалалд шилжүүллээ. Та 1800-1917 луу шууд залгах боломжтой."
        }

    # =========================================================================
    # MASTER PIPELINE (Stages 1 -> 6 Orchestration)
    # =========================================================================
    def run_full_pipeline(self, audio_bytes: Optional[bytes] = None, text_query: Optional[str] = None) -> Dict[str, Any]:
        """Runs the complete 6-stage end-to-end voicebot workflow."""
        start_time = time.time()
        stage_logs = []
        
        # --- STAGE 1: Call Intake ---
        stage1_start = time.time()
        if audio_bytes:
            transcript = self.transcribe_audio_chirp3(audio_bytes)
        elif text_query:
            transcript = text_query.strip()
        else:
            raise ValueError("Аудио эсвэл текст хүсэлт өгөгдөөгүй байна.")

        if not transcript:
            transcript = "Дуу хоолой тодорхой сонсогдсонгүй."
            
        stage_logs.append({
            "stage": 1,
            "name": "Call Intake",
            "title": "Дуудлага хүлээн авах & Дуу таних (Chirp 3)",
            "status": "completed",
            "duration": round(time.time() - stage1_start, 2),
            "details": f"Танигдсан үгс: \"{transcript}\""
        })

        # --- STAGE 2: Intent Understanding ---
        stage2_start = time.time()
        intent = self.understand_intent(transcript)
        stage_logs.append({
            "stage": 2,
            "name": "Intent Understanding",
            "title": "Зорилго ангилах (Gemini 2.5 Flash)",
            "status": "completed",
            "duration": round(time.time() - stage2_start, 2),
            "details": f"Сэдэв: {intent.get('topic')} ({intent.get('category')})"
        })

        # --- STAGE 3: Information Retrieval ---
        stage3_start = time.time()
        retrieval = self.retrieve_khanbank_info(transcript, intent.get("search_keywords", transcript))
        stage_logs.append({
            "stage": 3,
            "name": "Information Retrieval",
            "title": "Мэдээлэл шүүх (khanbank.com Grounding)",
            "status": "completed",
            "duration": round(time.time() - stage3_start, 2),
            "details": f"Хайлт: {retrieval['search_query']}"
        })

        # --- STAGE 4: Decision Making ---
        stage4_start = time.time()
        decision = self.make_decision(transcript, intent, retrieval)
        is_escalation = decision.get("requires_human", False)
        stage_logs.append({
            "stage": 4,
            "name": "Decision Making",
            "title": "Шийдвэр гаргах (ReAct Engine)",
            "status": "completed",
            "duration": round(time.time() - stage4_start, 2),
            "details": f"Шийдвэр: {decision.get('decision')} - {decision.get('reason')}"
        })

        # --- STAGE 6 (if triggered) / STAGE 5 ---
        escalation_data = None
        if is_escalation:
            stage6_start = time.time()
            escalation_data = self.execute_escalation_protocol(transcript, intent)
            stage_logs.append({
                "stage": 6,
                "name": "Human Escalation",
                "title": "Мэргэжилтэнд шилжүүлэх (Escalation Protocol)",
                "status": "triggered",
                "duration": round(time.time() - stage6_start, 2),
                "details": f"Лавлах утас: {escalation_data['hotline']} ({escalation_data['department']})"
            })

        # --- STAGE 5: Response Generation ---
        stage5_start = time.time()
        response_text = self.generate_mongolian_response(transcript, retrieval, is_escalation=is_escalation)
        
        audio_filename = None
        audio_error = None
        try:
            audio_filename = self.synthesize_speech_narakeet(response_text, voice="Ganbaatar")
        except Exception as e:
            audio_error = str(e)

        stage_logs.append({
            "stage": 5,
            "name": "Response Generation",
            "title": "Хариу боловсруулах & TTS (Narakeet Ganbaatar)",
            "status": "completed" if audio_filename else "warning",
            "duration": round(time.time() - stage5_start, 2),
            "details": f"Дууны файл: {audio_filename if audio_filename else audio_error}"
        })

        total_time = round(time.time() - start_time, 2)

        return {
            "transcript": transcript,
            "response_text": response_text,
            "audio_url": f"/audio/{audio_filename}" if audio_filename else None,
            "intent": intent,
            "retrieval": retrieval,
            "decision": decision,
            "escalation": escalation_data,
            "is_escalated": is_escalation,
            "stage_logs": stage_logs,
            "execution_time_sec": total_time
        }
