#!/usr/bin/env python3
"""
RAG System Unified Launcher
===========================

A comprehensive launcher that starts all RAG system components:
- Ollama server
- RAG API server (port 8001)
- Backend server (port 8000)  
- Frontend server (port 3000)

Features:
- Single command startup
- Real-time log aggregation
- Process health monitoring
- Graceful shutdown
- Production-ready deployment support

Usage:
    python run_system.py [--mode dev|prod] [--logs-only] [--no-frontend]
"""

import subprocess
import threading
import time
import signal
import sys
import os
import argparse
import json
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, TextIO
import logging
from dataclasses import dataclass
import psutil

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / ".venv"
VENV_PYTHON = VENV_DIR / "bin" / "python"
VENV_BIN = VENV_DIR / "bin"

@dataclass
class ServiceConfig:
    name: str
    command: List[str]
    port: int
    cwd: Optional[str] = None
    env: Optional[Dict[str, str]] = None
    health_check_path: str = "/health"
    startup_delay: int = 2
    required: bool = True

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels and services."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    
    SERVICE_COLORS = {
        'ollama': '\033[94m',     # Blue
        'rag-api': '\033[95m',    # Magenta
        'backend': '\033[96m',    # Cyan
        'frontend': '\033[93m',   # Yellow
        'system': '\033[92m',     # Green
    }
    
    RESET = '\033[0m'
    
    def format(self, record):
        # Add service-specific coloring
        service_name = getattr(record, 'service', 'system')
        service_color = self.SERVICE_COLORS.get(service_name, self.COLORS.get(record.levelname, ''))
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
        
        # Create colored log line
        colored_service = f"{service_color}[{service_name.upper()}]{self.RESET}"
        colored_level = f"{self.COLORS.get(record.levelname, '')}{record.levelname}{self.RESET}"
        
        return f"{timestamp} {colored_service} {colored_level}: {record.getMessage()}"

class ServiceManager:
    """Manages multiple system services with logging and health monitoring."""
    
    def __init__(self, mode: str = "dev", logs_dir: str = "logs"):
        self.mode = mode
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(exist_ok=True)
        
        self.processes: Dict[str, subprocess.Popen] = {}
        self.log_threads: Dict[str, threading.Thread] = {}
        self.running = False
        
        # Setup logging
        self.setup_logging()
        
        # Service configurations
        self.services = self._get_service_configs()
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def setup_logging(self):
        """Setup centralized logging with colors."""
        # Create main logger
        self.logger = logging.getLogger('system')
        self.logger.setLevel(logging.INFO)
        
        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ColoredFormatter())
        self.logger.addHandler(console_handler)
        
        # File handler for system logs
        file_handler = logging.FileHandler(self.logs_dir / 'system.log')
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s'
        ))
        self.logger.addHandler(file_handler)
    
    def _get_service_configs(self) -> Dict[str, ServiceConfig]:
        """Define service configurations based on mode."""
        python_executable = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable
        base_env = {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
        if VENV_DIR.exists():
            base_env["VIRTUAL_ENV"] = str(VENV_DIR)
            base_env["PATH"] = f"{VENV_BIN}{os.pathsep}{os.environ.get('PATH', '')}"

        base_configs = {
            'ollama': ServiceConfig(
                name='ollama',
                command=['ollama', 'serve'],
                port=11434,
                health_check_path="/api/tags",
                startup_delay=5,
                required=True
            ),
            'rag-api': ServiceConfig(
                name='rag-api',
                command=[python_executable, '-m', 'rag_system.api_server'],
                port=8001,
                cwd=str(PROJECT_ROOT),
                env=base_env,
                health_check_path="/models",
                startup_delay=3,
                required=True
            ),
            'backend': ServiceConfig(
                name='backend',
                command=[python_executable, 'backend/server.py'],
                port=8000,
                cwd=str(PROJECT_ROOT),
                env=base_env,
                startup_delay=2,
                required=True
            ),
            'frontend': ServiceConfig(
                name='frontend',
                command=['npm', 'run', 'dev' if self.mode == 'dev' else 'start'],
                port=3000,
                cwd=str(PROJECT_ROOT),
                health_check_path="/",
                startup_delay=5,
                required=False  # Optional in case Node.js not available
            )
        }
        
        # Production mode adjustments
        if self.mode == 'prod':
            # Use production build for frontend
            base_configs['frontend'].command = ['npm', 'run', 'start']
            # Add production environment variables
            base_configs['rag-api'].env = {**base_env, 'NODE_ENV': 'production'}
            base_configs['backend'].env = {**base_env, 'NODE_ENV': 'production'}
        
        return base_configs
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown()
        sys.exit(0)
    
    def is_port_in_use(self, port: int) -> bool:
        """Check if a port is already in use."""
        try:
            for conn in psutil.net_connections():
                if conn.laddr.port == port and conn.status == 'LISTEN':
                    return True
            return False
        except (psutil.AccessDenied, AttributeError):
            # Fallback method
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(('localhost', port)) == 0

    def find_port_listeners(self, port: int):
        """Return psutil processes listening on a TCP port."""
        listeners = []
        try:
            for conn in psutil.net_connections(kind="tcp"):
                if conn.status != "LISTEN" or not conn.laddr or conn.laddr.port != port or not conn.pid:
                    continue
                try:
                    listeners.append(psutil.Process(conn.pid))
                except psutil.Error:
                    continue
        except psutil.Error:
            pass
        if listeners:
            return listeners

        try:
            result = subprocess.run(
                ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            for raw_pid in result.stdout.splitlines():
                try:
                    listeners.append(psutil.Process(int(raw_pid.strip())))
                except (ValueError, psutil.Error):
                    continue
        except (subprocess.SubprocessError, FileNotFoundError):
            return []
        return listeners
    
    def check_prerequisites(self) -> bool:
        """Check if all required tools are available."""
        self.logger.info("🔍 Checking prerequisites...")
        
        missing_tools = []

        if not VENV_PYTHON.exists():
            missing_tools.append(".venv/bin/python (run: python3 -m venv .venv && source .venv/bin/activate && python -m pip install -r backend/requirements.txt -r rag_system/requirements.txt)")
        else:
            try:
                self._check_python_environment()
            except RuntimeError:
                return False
        
        # Check Ollama
        if not self._command_exists('ollama'):
            missing_tools.append('ollama (https://ollama.ai)')
        
        # Check Python
        if not self._command_exists('python') and not self._command_exists('python3'):
            missing_tools.append('python')
        
        # Check Node.js (optional)
        if not self._command_exists('npm'):
            self.logger.warning("⚠️  npm not found - frontend will be disabled")
            self.services['frontend'].required = False
        
        if missing_tools:
            self.logger.error(f"❌ Missing required tools: {', '.join(missing_tools)}")
            return False
        
        self.logger.info("✅ All prerequisites satisfied")
        return True

    def _check_python_environment(self):
        """Verify that required Python packages are installed in the project venv."""
        imports = ["fastapi", "uvicorn", "python_multipart", "transformers", "torch", "lancedb", "docling"]
        probe = "import " + ", ".join(imports)
        result = subprocess.run(
            [str(VENV_PYTHON), "-c", probe],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            self.logger.error("❌ Project .venv is missing required Python packages")
            self.logger.error("   Run: source .venv/bin/activate")
            self.logger.error("   Then: python -m pip install -r backend/requirements.txt")
            self.logger.error("   And:  python -m pip install -r rag_system/requirements.txt")
            raise RuntimeError(result.stderr.strip() or "Python dependency probe failed")

        if Path(sys.executable).resolve() != VENV_PYTHON.resolve():
            self.logger.warning(f"⚠️  Launcher is running under {sys.executable}")
            self.logger.warning(f"   Services will still use project venv: {VENV_PYTHON}")
    
    def _command_exists(self, command: str) -> bool:
        """Check if a command exists in PATH."""
        try:
            subprocess.run([command, '--version'], 
                         capture_output=True, check=True, timeout=5)
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def ensure_models(self):
        """Ensure required Ollama models are available."""
        self.logger.info("📥 Checking required models...")
        
        required_models = ['qwen3:8b']
        
        try:
            # Get list of installed models
            result = subprocess.run(['ollama', 'list'], 
                                  capture_output=True, text=True, timeout=10)
            installed_models = result.stdout
            
            for model in required_models:
                if model not in installed_models:
                    self.logger.info(f"📥 Pulling {model}...")
                    subprocess.run(['ollama', 'pull', model], 
                                 check=True, timeout=300)  # 5 min timeout
                    self.logger.info(f"✅ {model} ready")
                else:
                    self.logger.info(f"✅ {model} already available")
                    
        except subprocess.TimeoutExpired:
            self.logger.warning("⚠️  Model check timed out - continuing anyway")
        except subprocess.CalledProcessError as e:
            self.logger.warning(f"⚠️  Could not check/pull models: {e}")
    
    def start_service(self, service_name: str, config: ServiceConfig) -> bool:
        """Start a single service."""
        if service_name in self.processes:
            self.logger.warning(f"⚠️  {service_name} already running")
            return True
        
        # Check if port is in use
        if self.is_port_in_use(config.port):
            if self.health_check(service_name, config):
                self.logger.info(f"✅ {service_name} already running on port {config.port}")
                return True
            listeners = self.find_port_listeners(config.port)
            owner = ", ".join(f"{p.pid}:{p.name()}" for p in listeners) or "unknown process"
            self.logger.error(f"❌ Port {config.port} is in use by {owner}, but {service_name} health check failed")
            self.logger.error("   Run './start-localgpt --stop' to clear stale LocalGPT ports, then start again.")
            return False
        
        self.logger.info(f"🔄 Starting {service_name} on port {config.port}...")
        
        try:
            # Setup environment
            env = os.environ.copy()
            if config.env:
                env.update(config.env)
            
            # Start process
            process = subprocess.Popen(
                config.command,
                cwd=config.cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            self.processes[service_name] = process

            # Start log monitoring thread
            log_thread = threading.Thread(
                target=self._monitor_service_logs,
                args=(service_name, process),
                daemon=True
            )
            log_thread.start()
            self.log_threads[service_name] = log_thread

            # Poll immediately — no fixed sleep. _wait_for_health retries every
            # second until the service responds or the timeout expires.
            if process.poll() is None:
                if self._wait_for_health(service_name, config):
                    self.logger.info(f"✅ {service_name} started successfully (PID: {process.pid})")
                    return True
                self.logger.error(f"❌ {service_name} started but failed health check")
                return False
            else:
                self.logger.error(f"❌ {service_name} failed to start")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Failed to start {service_name}: {e}")
            return False
    
    def _monitor_service_logs(self, service_name: str, process: subprocess.Popen):
        """Monitor service logs and forward to main logger."""
        service_logger = logging.getLogger(service_name)
        service_logger.setLevel(logging.INFO)
        
        # Add file handler for this service
        file_handler = logging.FileHandler(self.logs_dir / f'{service_name}.log')
        file_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        service_logger.addHandler(file_handler)
        
        try:
            for line in iter(process.stdout.readline, ''):
                if line.strip():
                    # Create log record with service context
                    record = logging.LogRecord(
                        name=service_name,
                        level=logging.INFO,
                        pathname='',
                        lineno=0,
                        msg=line.strip(),
                        args=(),
                        exc_info=None
                    )
                    record.service = service_name
                    
                    # Log to both service file and main console
                    service_logger.handle(record)
                    self.logger.handle(record)
                    
        except Exception as e:
            self.logger.error(f"Error monitoring {service_name} logs: {e}")
    
    def health_check(self, service_name: str, config: ServiceConfig) -> bool:
        """Perform health check on a service."""
        try:
            url = f"http://localhost:{config.port}{config.health_check_path}"
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except:
            return False

    def _wait_for_health(self, service_name: str, config: ServiceConfig, timeout: int = 30) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if service_name in self.processes and self.processes[service_name].poll() is not None:
                return False
            if self.health_check(service_name, config):
                return True
            time.sleep(1)
        return False
    
    def start_all(self, skip_frontend: bool = False) -> bool:
        """Start all services. Ollama must be ready first; everything else starts in parallel."""
        self.logger.info("🚀 Starting RAG System Components...")

        if not self.check_prerequisites():
            return False

        self.running = True

        # Ollama is a hard prerequisite — start it synchronously first.
        if not self._start_ollama():
            self.logger.error("❌ Failed to start required service: ollama")
            return False

        # rag-api, backend, and frontend have no startup dependency on each other,
        # so launch them concurrently and wait for all to report healthy.
        parallel_services = [s for s in ('rag-api', 'backend') if s in self.services]
        if not skip_frontend and 'frontend' in self.services:
            parallel_services.append('frontend')

        results: Dict[str, bool] = {}

        def _start_one(name: str) -> None:
            results[name] = self.start_service(name, self.services[name])

        threads = [threading.Thread(target=_start_one, args=(name,), daemon=True)
                   for name in parallel_services]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        failed_services = [
            name for name in parallel_services
            if not results.get(name) and self.services[name].required
        ]
        for name in parallel_services:
            if not results.get(name) and not self.services[name].required:
                self.logger.warning(f"⚠️  Optional service failed to start: {name}")

        if failed_services:
            self.logger.error(f"❌ Failed to start required services: {', '.join(failed_services)}")
            return False

        self._print_status_summary()
        return True
    
    def _start_ollama(self) -> bool:
        """Special handling for Ollama startup."""
        # Check if Ollama is already running
        if self.is_port_in_use(11434):
            self.logger.info("✅ Ollama already running")
            self.ensure_models()
            return True
        
        # Start Ollama
        if self.start_service('ollama', self.services['ollama']):
            self.ensure_models()
            return True
        
        return False
    
    def _print_status_summary(self, title: str = "RAG System Started!"):
        """Print system status summary."""
        self.logger.info("")
        self.logger.info(f"🎉 {title}")
        self.logger.info("📊 Services Status:")
        
        for service_name, config in self.services.items():
            if self.health_check(service_name, config):
                status = "✅ Healthy"
                url = f"http://localhost:{config.port}"
                self.logger.info(f"   • {service_name.capitalize():<10}: {status:<10} {url}")
            elif service_name in self.processes or self.is_port_in_use(config.port):
                listeners = self.find_port_listeners(config.port)
                owner = ", ".join(f"{p.pid}:{p.name()}" for p in listeners) or "unknown process"
                status = f"⚠️  Listening, unhealthy ({owner})"
                url = f"http://localhost:{config.port}"
                self.logger.info(f"   • {service_name.capitalize():<10}: {status:<10} {url}")
            else:
                self.logger.info(f"   • {service_name.capitalize():<10}: ❌ Stopped")
        
        self.logger.info("")
        self.logger.info("🌐 Access your RAG system at: http://localhost:3000")
        self.logger.info("")
        self.logger.info("📋 Useful commands:")
        self.logger.info("   • Stop system:  Ctrl+C")
        self.logger.info("   • Check logs:   tail -f logs/*.log")
        self.logger.info("   • Health check: ./start-localgpt --health")
    
    def shutdown(self):
        """Gracefully shutdown all services."""
        if not self.running:
            return
        
        self.logger.info("🛑 Shutting down RAG system...")
        self.running = False
        
        # Stop services in reverse order
        for service_name in reversed(list(self.processes.keys())):
            self._stop_service(service_name)
        
        self.logger.info("✅ All services stopped")
    
    def _stop_service(self, service_name: str):
        """Stop a single service."""
        if service_name not in self.processes:
            return
        
        process = self.processes[service_name]
        self.logger.info(f"🔄 Stopping {service_name}...")
        
        try:
            # Try graceful shutdown first
            process.terminate()
            
            # Wait up to 10 seconds for graceful shutdown
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown fails
                process.kill()
                process.wait()
            
            self.logger.info(f"✅ {service_name} stopped")
            
        except Exception as e:
            self.logger.error(f"❌ Error stopping {service_name}: {e}")
        finally:
            del self.processes[service_name]

    def stop_port_listeners(self, include_ollama: bool = False):
        """Stop processes listening on LocalGPT service ports."""
        service_names = list(reversed(list(self.services.keys())))
        for service_name in service_names:
            if service_name == "ollama" and not include_ollama:
                continue
            config = self.services[service_name]
            for process in self.find_port_listeners(config.port):
                proc_name = process.name()
                self.logger.info(f"🔄 Stopping {service_name} listener {process.pid}:{proc_name} on port {config.port}...")
                try:
                    process.terminate()
                    try:
                        process.wait(timeout=8)
                    except psutil.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                    self.logger.info(f"✅ Stopped {process.pid}:{proc_name}")
                except psutil.Error as e:
                    self.logger.warning(f"⚠️  Could not stop process on port {config.port}: {e}")
    
    def monitor(self):
        """Monitor running services and restart if needed."""
        self.logger.info("👁️  Monitoring services... (Press Ctrl+C to stop)")
        
        try:
            while self.running:
                time.sleep(30)  # Check every 30 seconds
                
                for service_name, process in list(self.processes.items()):
                    if process.poll() is not None:
                        self.logger.warning(f"⚠️  {service_name} has stopped unexpectedly")
                        
                        # Restart the service
                        config = self.services[service_name]
                        if config.required:
                            self.logger.info(f"🔄 Restarting {service_name}...")
                            del self.processes[service_name]
                            self.start_service(service_name, config)
                        
        except KeyboardInterrupt:
            self.logger.info("Monitoring stopped by user")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='RAG System Unified Launcher')
    parser.add_argument('--mode', choices=['dev', 'prod'], default='dev',
                       help='Run mode (default: dev)')
    parser.add_argument('--logs-only', action='store_true',
                       help='Only show aggregated logs from running services')
    parser.add_argument('--no-frontend', action='store_true',
                       help='Skip frontend startup')
    parser.add_argument('--health', action='store_true',
                       help='Check health of running services')
    parser.add_argument('--stop', action='store_true',
                       help='Stop LocalGPT frontend/backend/RAG listeners')
    parser.add_argument('--stop-ollama', action='store_true',
                       help='Also stop Ollama when used with --stop')
    
    args = parser.parse_args()
    
    # Create service manager
    manager = ServiceManager(mode=args.mode)
    
    try:
        if args.health:
            # Health check mode
            manager._print_status_summary("RAG System Status")
            return
        
        if args.stop:
            manager.logger.info("🛑 Stopping LocalGPT service ports...")
            manager.stop_port_listeners(include_ollama=args.stop_ollama)
            manager._print_status_summary("RAG System Status")
            return
        
        if args.logs_only:
            # Logs only mode - just tail existing logs
            manager.logger.info("📋 Showing aggregated logs... (Press Ctrl+C to stop)")
            manager.monitor()
            return
        
        # Normal startup mode
        if manager.start_all(skip_frontend=args.no_frontend):
            manager.monitor()
        else:
            manager.logger.error("❌ System startup failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        manager.logger.info("Received interrupt signal")
    finally:
        manager.shutdown()

if __name__ == "__main__":
    main() 
