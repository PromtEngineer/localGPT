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
    python run_system.py --health
    python run_system.py --stop
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

GENERATION_MODEL = os.getenv("GENERATION_MODEL", "qwen3.5:9b")
ENRICHMENT_MODEL = os.getenv("ENRICHMENT_MODEL", "qwen3.5:4b")


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
    build_command: Optional[List[str]] = None

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
        self.pidfile = self.logs_dir / 'run_system.pid'

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
        base_configs = {
            'ollama': ServiceConfig(
                name='ollama',
                command=['ollama', 'serve'],
                port=11434,
                health_check_path='/api/tags',
                startup_delay=5,
                required=True
            ),
            'rag-api': ServiceConfig(
                name='rag-api',
                command=[sys.executable, '-m', 'rag_system.api_server'],
                port=8001,
                startup_delay=3,
                required=True
            ),
            'backend': ServiceConfig(
                name='backend',
                command=[sys.executable, 'backend/server.py'],
                port=8000,
                startup_delay=2,
                required=True
            ),
            'frontend': ServiceConfig(
                name='frontend',
                command=['npm', 'run', 'dev' if self.mode == 'dev' else 'start'],
                port=3000,
                health_check_path='/',
                startup_delay=5,
                required=False  # Optional in case Node.js not available
            )
        }

        # Production mode adjustments
        if self.mode == 'prod':
            # `next start` needs an existing .next build, so build before starting
            base_configs['frontend'].command = ['npm', 'run', 'start']
            base_configs['frontend'].build_command = ['npm', 'run', 'build']
            # Add production environment variables
            base_configs['rag-api'].env = {'NODE_ENV': 'production'}
            base_configs['backend'].env = {'NODE_ENV': 'production'}

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
    
    def check_prerequisites(self) -> bool:
        """Check if all required tools are available."""
        self.logger.info("🔍 Checking prerequisites...")
        
        missing_tools = []
        
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
        
        required_models = [GENERATION_MODEL, ENRICHMENT_MODEL]

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
            self.logger.warning(f"⚠️  Port {config.port} already in use, skipping {service_name}")
            return not config.required
        
        if config.build_command and not self._run_build(service_name, config):
            return False

        self.logger.info(f"🔄 Starting {service_name} on port {config.port}...")

        try:
            # Setup environment
            env = os.environ.copy()
            if config.env:
                env.update(config.env)

            # Start process in its own session/process group (POSIX; ignored on
            # Windows) so _stop_service can signal the whole tree — wrappers
            # like npm spawn the real server as a child.
            process = subprocess.Popen(
                config.command,
                cwd=config.cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                start_new_session=True
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
            
            # Wait for startup
            time.sleep(config.startup_delay)
            
            # Check if process is still running
            if process.poll() is None:
                self.logger.info(f"✅ {service_name} started successfully (PID: {process.pid})")
                self._write_pidfile()
                return True
            else:
                self.logger.error(f"❌ {service_name} failed to start")
                del self.processes[service_name]
                return False

        except Exception as e:
            self.logger.error(f"❌ Failed to start {service_name}: {e}")
            self.processes.pop(service_name, None)
            return False

    def _run_build(self, service_name: str, config: ServiceConfig) -> bool:
        """Run a service's build step before starting it (prod frontend needs `next build`)."""
        self.logger.info(f"🏗️  Building {service_name}: {' '.join(config.build_command)}")
        try:
            result = subprocess.run(config.build_command, cwd=config.cwd)
        except FileNotFoundError as e:
            self.logger.error(f"❌ Build command for {service_name} not found: {e}")
            return False
        if result.returncode != 0:
            self.logger.error(f"❌ Build failed for {service_name} (exit {result.returncode})")
            return False
        self.logger.info(f"✅ {service_name} build complete")
        return True

    def _write_pidfile(self):
        """Persist launcher + child PIDs so `--stop` can find them in another shell."""
        data = {
            'launcher': os.getpid(),
            'services': {name: proc.pid for name, proc in self.processes.items()},
        }
        try:
            self.pidfile.write_text(json.dumps(data, indent=2))
        except OSError as e:
            self.logger.warning(f"⚠️  Could not write pidfile {self.pidfile}: {e}")

    def _read_pidfile(self) -> Dict:
        try:
            return json.loads(self.pidfile.read_text())
        except (OSError, ValueError):
            return {}

    def _clear_pidfile(self):
        try:
            self.pidfile.unlink()
        except OSError:
            pass


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
        """Perform an HTTP health check against a service."""
        url = f"http://localhost:{config.port}{config.health_check_path}"
        try:
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def print_health_report(self) -> bool:
        """Run a real health check per service and report pass/fail."""
        self.logger.info("🏥 Health check:")
        all_required_healthy = True

        for service_name, config in self.services.items():
            url = f"http://localhost:{config.port}{config.health_check_path}"
            if not self.is_port_in_use(config.port):
                status = "❌ Not running"
                healthy = False
            elif self.health_check(service_name, config):
                status = "✅ Healthy"
                healthy = True
            else:
                status = "⚠️  Port open, health check failed"
                healthy = False

            self.logger.info(f"   • {service_name.capitalize():<10}: {status:<32} {url}")
            if config.required and not healthy:
                all_required_healthy = False

        self.logger.info("")
        if all_required_healthy:
            self.logger.info("✅ All required services are healthy")
        else:
            self.logger.error("❌ One or more required services are unhealthy")
        return all_required_healthy


    def start_all(self, skip_frontend: bool = False) -> bool:
        """Start all services in order."""
        self.logger.info("🚀 Starting RAG System Components...")
        
        if not self.check_prerequisites():
            return False
        
        self.running = True
        failed_services = []
        
        # Start services in dependency order
        service_order = ['ollama', 'rag-api', 'backend']
        if not skip_frontend and 'frontend' in self.services:
            service_order.append('frontend')
        
        for service_name in service_order:
            if service_name not in self.services:
                continue
                
            config = self.services[service_name]
            
            # Special handling for Ollama
            if service_name == 'ollama':
                if not self._start_ollama():
                    if config.required:
                        failed_services.append(service_name)
                        continue
                    else:
                        self.logger.warning(f"⚠️  Skipping optional service: {service_name}")
                        continue
            else:
                if not self.start_service(service_name, config):
                    if config.required:
                        failed_services.append(service_name)
                    else:
                        self.logger.warning(f"⚠️  Skipping optional service: {service_name}")
        
        if failed_services:
            self.logger.error(f"❌ Failed to start required services: {', '.join(failed_services)}")
            return False
        
        # Print status summary
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
    
    def _print_status_summary(self):
        """Print system status summary."""
        self.logger.info("")
        self.logger.info("🎉 RAG System Started!")
        self.logger.info("📊 Services Status:")
        
        for service_name, config in self.services.items():
            if service_name in self.processes or self.is_port_in_use(config.port):
                status = "✅ Running"
                url = f"http://localhost:{config.port}"
                self.logger.info(f"   • {service_name.capitalize():<10}: {status:<10} {url}")
            else:
                self.logger.info(f"   • {service_name.capitalize():<10}: ❌ Stopped")
        
        self.logger.info("")
        self.logger.info("🌐 Access your RAG system at: http://localhost:3000")
        self.logger.info("")
        self.logger.info("📋 Useful commands:")
        self.logger.info("   • Stop system:  Ctrl+C  (or: python run_system.py --stop)")
        self.logger.info("   • Check logs:   tail -f logs/*.log")
        self.logger.info("   • Health check: python run_system.py --health")

    def stop_all(self) -> bool:
        """Stop services recorded in the pidfile (works from a separate shell)."""
        record = self._read_pidfile()
        if not record:
            self.logger.warning(f"⚠️  No pidfile at {self.pidfile} – nothing to stop.")
            return False

        stopped = 0

        # Stop the launcher first so its monitor loop cannot restart anything
        launcher_pid = record.get('launcher')
        if launcher_pid and launcher_pid != os.getpid():
            self._terminate_pid('launcher', launcher_pid)

        for service_name, pid in (record.get('services') or {}).items():
            if self._terminate_pid(service_name, pid):
                stopped += 1

        self._clear_pidfile()
        self.logger.info(f"✅ Stopped {stopped} service(s)")
        return stopped > 0

    def _terminate_pid(self, label: str, pid: int) -> bool:
        """Terminate a PID, escalating to kill, and reap its children."""
        try:
            process = psutil.Process(pid)
        except psutil.NoSuchProcess:
            self.logger.info(f"   • {label}: already stopped (pid {pid})")
            return False
        except psutil.Error as e:
            self.logger.warning(f"⚠️  Could not inspect {label} (pid {pid}): {e}")
            return False

        targets = [process]
        try:
            targets.extend(process.children(recursive=True))
        except psutil.Error:
            pass

        for target in reversed(targets):
            try:
                target.terminate()
            except psutil.Error:
                pass

        _, alive = psutil.wait_procs(targets, timeout=10)
        for target in alive:
            try:
                target.kill()
            except psutil.Error:
                pass

        self.logger.info(f"   • {label}: stopped (pid {pid})")
        return True

    def tail_logs(self):
        """Follow every logs/*.log file, like `tail -f logs/*.log`."""
        log_files = sorted(self.logs_dir.glob('*.log'))
        if not log_files:
            self.logger.warning(f"⚠️  No log files in {self.logs_dir}/ – start the system first.")
            return

        self.logger.info(f"📋 Following {len(log_files)} log file(s): {', '.join(f.name for f in log_files)}")
        handles = {}
        try:
            for path in log_files:
                handle = path.open('r', errors='replace')
                handle.seek(0, os.SEEK_END)
                handles[path.stem] = handle

            while True:
                emitted = False
                for name, handle in handles.items():
                    line = handle.readline()
                    while line:
                        print(f"[{name.upper()}] {line.rstrip()}")
                        emitted = True
                        line = handle.readline()
                if not emitted:
                    time.sleep(0.5)
        except KeyboardInterrupt:
            self.logger.info("Log tailing stopped by user")
        finally:
            for handle in handles.values():
                handle.close()

    def shutdown(self):
        """Gracefully shutdown all services."""
        if not self.running:
            return

        self.logger.info("🛑 Shutting down RAG system...")
        self.running = False

        # Stop services in reverse order
        for service_name in reversed(list(self.processes.keys())):
            self._stop_service(service_name)

        self._clear_pidfile()
        self.logger.info("✅ All services stopped")


    def _signal_process_tree(self, process: subprocess.Popen, sig: int):
        """Signal the service's whole process group (POSIX), else just the child."""
        if os.name == 'posix':
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass  # process group already gone
        elif sig == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()

    def _stop_service(self, service_name: str):
        """Stop a single service."""
        if service_name not in self.processes:
            return

        process = self.processes[service_name]
        self.logger.info(f"🔄 Stopping {service_name}...")

        try:
            # Try graceful shutdown first. Services are spawned with
            # start_new_session=True, so signal the process GROUP: terminate()
            # alone would kill only the direct child (e.g. npm) and orphan the
            # real server it spawned.
            self._signal_process_tree(process, signal.SIGTERM)

            # Wait up to 10 seconds for graceful shutdown
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown fails
                self._signal_process_tree(process, signal.SIGKILL)
                process.wait()

            self.logger.info(f"✅ {service_name} stopped")

        except Exception as e:
            self.logger.error(f"❌ Error stopping {service_name}: {e}")
        finally:
            del self.processes[service_name]
    
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
                       help='Stop all running services')
    
    args = parser.parse_args()
    
    # Create service manager
    manager = ServiceManager(mode=args.mode)
    
    try:
        if args.health:
            # Health check mode - real HTTP checks against each /health endpoint
            sys.exit(0 if manager.print_health_report() else 1)

        if args.stop:
            # Stop mode - terminate the processes recorded in the pidfile
            manager.logger.info("🛑 Stopping all RAG system processes...")
            sys.exit(0 if manager.stop_all() else 1)

        if args.logs_only:
            # Logs only mode - tail the log files written by a running launcher
            manager.logger.info("📋 Showing aggregated logs... (Press Ctrl+C to stop)")
            manager.tail_logs()
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