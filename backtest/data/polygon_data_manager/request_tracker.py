"""Track and report on all data requests"""
import json
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional

from .models import DataRequest

logger = logging.getLogger(__name__)


class RequestTracker:
    """Tracks all data requests for reporting and analysis"""
    
    def __init__(self, report_dir: Optional[Path] = None):
        self.report_dir = report_dir or Path.cwd() / 'temp'
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        self.all_requests: List[DataRequest] = []
        self.requests_by_plugin = defaultdict(list)
        self.current_plugin = "Unknown"
        
    def set_current_plugin(self, plugin_name: str):
        """Set the current plugin making requests"""
        self.current_plugin = plugin_name
        logger.info(f"Current plugin set to: {plugin_name}")
        
    def track_request(self, request: DataRequest):
        """Track a data request"""
        self.all_requests.append(request)
        self.requests_by_plugin[request.plugin_name].append(request)
        
    def generate_report(self) -> Tuple[str, str]:
        """Generate comprehensive data request report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create detailed JSON report
        json_report = self._create_json_report()
        
        # Save JSON report
        json_file = self.report_dir / f"polygon_data_report_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(json_report, f, indent=2)
        
        # Create human-readable summary
        summary_file = self.report_dir / f"polygon_data_report_{timestamp}_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(self._create_summary_report(json_report))
        
        logger.info(f"Reports generated: {json_file}, {summary_file}")
        return str(json_file), str(summary_file)
        
    def _create_json_report(self) -> Dict[str, Any]:
        """Create detailed JSON report structure"""
        total_api_calls = sum(1 for r in self.all_requests if r.source == 'polygon_api')
        total_cache_hits = sum(1 for r in self.all_requests if r.source != 'polygon_api')
        
        report = {
            "report_timestamp": datetime.now().isoformat(),
            "session_stats": {
                "total_requests": len(self.all_requests),
                "successful_requests": sum(1 for r in self.all_requests if r.success),
                "failed_requests": sum(1 for r in self.all_requests if not r.success),
                "api_calls": total_api_calls,
                "cache_hits": total_cache_hits,
                "total_processing_time_ms": sum(r.processing_time_ms for r in self.all_requests)
            },
            "by_plugin": {},
            "all_requests": []
        }
        
        # Group by plugin
        for plugin_name, requests in self.requests_by_plugin.items():
            plugin_stats = {
                "total_requests": len(requests),
                "successful": sum(1 for r in requests if r.success),
                "failed": sum(1 for r in requests if not r.success),
                "by_data_type": {
                    "bars": sum(1 for r in requests if r.data_type == 'bars'),
                    "trades": sum(1 for r in requests if r.data_type == 'trades'),
                    "quotes": sum(1 for r in requests if r.data_type == 'quotes')
                },
                "requests": []
            }
            
            for req in requests:
                req_data = {
                    "request_id": req.request_id,
                    "symbol": req.symbol,
                    "data_type": req.data_type,
                    "timeframe": req.timeframe,
                    "requested_start": req.start_time.isoformat(),
                    "requested_end": req.end_time.isoformat(),
                    "source": req.source,
                    "success": req.success,
                    "returned_count": req.returned_count,
                    "processing_time_ms": req.processing_time_ms
                }
                
                if req.actual_start:
                    req_data["actual_start"] = req.actual_start.isoformat()
                if req.actual_end:
                    req_data["actual_end"] = req.actual_end.isoformat()
                if req.error:
                    req_data["error"] = req.error
                
                plugin_stats["requests"].append(req_data)
                report["all_requests"].append(req_data)
            
            report["by_plugin"][plugin_name] = plugin_stats
        
        return report
        
    def _create_summary_report(self, json_report: Dict[str, Any]) -> str:
        """Create human-readable summary"""
        lines = []
        lines.append("=" * 80)
        lines.append("POLYGON DATA MANAGER REPORT")
        lines.append("=" * 80)
        lines.append(f"\nGenerated: {datetime.now()}")
        lines.append(f"Report Location: {self.report_dir}\n")
        
        # Session overview
        stats = json_report['session_stats']
        lines.append("SESSION OVERVIEW:")
        lines.append("-" * 40)
        lines.append(f"Total Requests: {stats['total_requests']}")
        lines.append(f"Successful: {stats['successful_requests']}")
        lines.append(f"Failed: {stats['failed_requests']}")
        lines.append(f"API Calls Made: {stats['api_calls']}")
        lines.append(f"Cache Hits: {stats['cache_hits']}")
        
        cache_hit_rate = (stats['cache_hits'] / max(1, stats['total_requests'])) * 100
        lines.append(f"Cache Hit Rate: {cache_hit_rate:.1f}%")
        lines.append(f"Total Processing Time: {stats['total_processing_time_ms']:.1f}ms\n")
        
        # Plugin breakdown
        lines.append("PLUGIN BREAKDOWN:")
        lines.append("-" * 40)
        
        for plugin_name, plugin_stats in json_report["by_plugin"].items():
            lines.append(f"\n{plugin_name}:")
            lines.append(f"  Total Requests: {plugin_stats['total_requests']}")
            lines.append(f"  Successful: {plugin_stats['successful']}")
            lines.append(f"  Failed: {plugin_stats['failed']}")
            
            # Show failed requests details
            failed_requests = [r for r in plugin_stats['requests'] if not r['success']]
            if failed_requests:
                lines.append(f"  FAILED REQUESTS:")
                for req in failed_requests[:3]:  # Show first 3
                    lines.append(f"    - {req['data_type']} for {req['symbol']}: "
                               f"{req.get('error', 'Unknown error')}")
        
        return "\n".join(lines)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get current tracking statistics"""
        total_api_calls = sum(1 for r in self.all_requests if r.source == 'polygon_api')
        total_cache_hits = sum(1 for r in self.all_requests if r.source != 'polygon_api')
        
        return {
            'total_requests': len(self.all_requests),
            'api_calls': total_api_calls,
            'cache_hits': total_cache_hits,
            'plugins_tracked': len(self.requests_by_plugin)
        }