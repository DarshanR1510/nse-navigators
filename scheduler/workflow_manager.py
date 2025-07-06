from trade_agents.orchestrator import AgentOrchestrator
from apscheduler.schedulers.background import BackgroundScheduler
import logging
import subprocess

class WorkflowManager:
    def __init__(self, orchestrator: AgentOrchestrator):
        """
        Initialize the WorkflowManager with an AgentOrchestrator instance.
        
        Args:
            orchestrator: An instance of AgentOrchestrator to manage trading agents.
        """
        self.orchestrator = orchestrator
        self.scheduler = BackgroundScheduler(timezone="Asia/Kolkata")

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
        
        self.schedule_daily_context()
        self.schedule_position_monitoring()
        self.schedule_market_context_update()
        self.scheduler.start()


    def schedule_daily_context(self):
        def run_daily_context():
            try:
                self.logger.info("Running daily context analysis...")
                self.orchestrator.run_daily_context_analysis()
                self.logger.info("Daily context analysis completed.")
            except Exception as e:
                self.logger.error(f"Error in daily context analysis: {e}")
        self.scheduler.add_job(run_daily_context, 'cron', hour=9, minute=10, id='daily_context_analysis')


    def schedule_position_monitoring(self):
        def run_position_monitoring():
            try:
                self.logger.info("Running position monitoring...")
                self.orchestrator.manage_position_monitoring()
                self.logger.info("Position monitoring completed.")
            except Exception as e:
                self.logger.error(f"Error in position monitoring: {e}")
        self.scheduler.add_job(run_position_monitoring, 'interval', minutes=60, id='position_monitoring')


    def schedule_market_context_update(self):
        def run_market_context_updater():
            try:
                self.logger.info("Running market context updater...")
                subprocess.run(["python", "-m", "market_tools.market_context_updater"])
                self.logger.info("Market context update completed.")
            except Exception as e:
                self.logger.error(f"Error in market context updater: {e}")
        self.scheduler.add_job(run_market_context_updater, 'cron', hour=9, minute=0, id='market_context_update')
