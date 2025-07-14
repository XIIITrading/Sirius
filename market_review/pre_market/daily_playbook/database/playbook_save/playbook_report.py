import os
import sys
from datetime import datetime, date
from pathlib import Path
import supabase
from supabase import create_client
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from PIL import Image as PILImage
import logging
from typing import Dict, List, Tuple
from dotenv import load_dotenv
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def find_project_root():
    """Find the project root by looking for .env file."""
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if (parent / '.env').exists():
            return parent
    return None

# Load environment variables
project_root = find_project_root()
if project_root:
    env_path = project_root / '.env'
    load_dotenv(env_path)
    logging.info(f"Loaded .env from: {env_path}")
else:
    load_dotenv()
    logging.warning("Could not find project root, attempting to load .env from current directory")

class DailyPlaybookReportGenerator:
    def __init__(self, supabase_url: str, supabase_key: str):
        """Initialize the report generator."""
        self.supabase = create_client(supabase_url, supabase_key)
        self.today = date.today()
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
    def _setup_custom_styles(self):
        """Set up custom PDF styles."""
        # Main title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            textColor=colors.HexColor('#1f2937'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        
        # Symbol header style
        self.styles.add(ParagraphStyle(
            name='SymbolHeader',
            parent=self.styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor('#2563eb'),
            spaceAfter=12,
            spaceBefore=20,
            alignment=TA_LEFT
        ))
        
        # Zone info style
        self.styles.add(ParagraphStyle(
            name='ZoneInfo',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#6b7280'),
            alignment=TA_CENTER
        ))
        
        # Statement style
        self.styles.add(ParagraphStyle(
            name='Statement',
            parent=self.styles['Normal'],
            fontSize=9,
            leading=11
        ))
        
    def fetch_today_plays(self) -> pd.DataFrame:
        """Fetch today's plays from Supabase."""
        try:
            result = self.supabase.table('daily_playbook').select("*").eq(
                'playbook_date', str(self.today)
            ).order('symbol').order('rank').execute()
            
            if not result.data:
                raise ValueError(f"No plays found for {self.today}")
                
            df = pd.DataFrame(result.data)
            logging.info(f"Fetched {len(df)} plays from database")
            return df
            
        except Exception as e:
            logging.error(f"Error fetching plays: {e}")
            raise
            
    def find_available_images(self, folder_path: Path) -> Dict[str, Path]:
        """Find all ticker PNG images in the folder."""
        images = {}
        for img_file in folder_path.glob("*.png"):
            ticker = img_file.stem.upper()
            if ticker and len(ticker) <= 10 and ticker.isalpha():
                images[ticker] = img_file
                logging.info(f"Found image for {ticker}")
        return images
        
    def resize_image_for_pdf(self, image_path: Path, max_width: float = 7*inch, max_height: float = 4*inch) -> Tuple[Path, float, float]:
        """Resize image to fit PDF while maintaining aspect ratio."""
        try:
            img = PILImage.open(image_path)
            width, height = img.size
            
            # Calculate scaling to fit within max dimensions
            width_scale = max_width / width
            height_scale = max_height / height
            scale = min(width_scale, height_scale)
            
            new_width = width * scale
            new_height = height * scale
            
            return image_path, new_width, new_height
        except Exception as e:
            logging.error(f"Error processing image {image_path}: {e}")
            return image_path, max_width, max_height
            
    def format_rr_ratio(self, value: float) -> str:
        """Format risk/reward ratio for display."""
        if value >= 1:
            return f"{value:.1f}:1"
        else:
            return f"1:{1/value:.1f}" if value > 0 else "N/A"
            
    def generate_pdf(self, df: pd.DataFrame, images: Dict[str, Path], output_path: Path):
        """Generate the PDF report with grids first, then images."""
        from reportlab.platypus import PageBreak, KeepTogether
        from reportlab.lib.pagesizes import landscape, letter
        
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=landscape(letter),
            rightMargin=50,
            leftMargin=50,
            topMargin=50,
            bottomMargin=30,
        )
        
        story = []
        
        # Title with date
        title_text = f"Daily Trading Playbook - {self.today.strftime('%B %d, %Y')}"
        title = Paragraph(title_text, self.styles['CustomTitle'])
        story.append(title)
        story.append(Spacer(1, 0.3*inch))
        
        # Summary statistics
        total_plays = len(df)
        unique_symbols = df['symbol'].nunique()
        avg_bull_rr = df['bullish_rr'].mean()
        avg_bear_rr = df['bearish_rr'].mean()
        
        summary_text = f"<b>Today's Summary:</b> {unique_symbols} symbols | {total_plays} total plays | Avg R:R - Bull: {avg_bull_rr:.2f}:1, Bear: {avg_bear_rr:.2f}:1"
        summary = Paragraph(summary_text, self.styles['Normal'])
        story.append(summary)
        story.append(Spacer(1, 0.5*inch))
        
        # Section 1: All Trading Grids
        section_header = Paragraph("Trading Zones & Entry Criteria", self.styles['CustomTitle'])
        story.append(section_header)
        story.append(Spacer(1, 0.3*inch))
        
        # Group plays by symbol
        grouped = df.groupby('symbol', sort=False)
        
        for symbol, group in grouped:
            # Symbol header
            symbol_header = Paragraph(f"{symbol}", self.styles['SymbolHeader'])
            story.append(symbol_header)
            
            # Zone range info
            price_lows = group['price_low'].values
            price_highs = group['price_high'].values
            zone_text = f"Zone Range: ${min(price_lows):.2f} - ${max(price_highs):.2f}"
            zone_info = Paragraph(zone_text, self.styles['ZoneInfo'])
            story.append(zone_info)
            story.append(Spacer(1, 0.1*inch))
            
            # Create detailed plays table
            table_data = [
                ['Rank', 'Zone', 'Entry Criteria', 'Target', 'Stop', 'R:R']
            ]
            
            for _, play in group.iterrows():
                # Calculate stop levels
                bullish_stop = play['price_low']
                bearish_stop = play['price_high']
                
                # Bullish play row
                table_data.append([
                    play['rank'].capitalize(),
                    f"${play['price_low']:.2f} - ${play['price_high']:.2f}",
                    Paragraph(f"<font color='green'><b>BULLISH:</b></font> {play['bullish_statement']}", self.styles['Statement']),
                    f"${play['bullish_target']:.2f}",
                    f"${bullish_stop:.2f}",
                    self.format_rr_ratio(play['bullish_rr'])
                ])
                
                # Bearish play row  
                table_data.append([
                    '',
                    '',
                    Paragraph(f"<font color='red'><b>BEARISH:</b></font> {play['bearish_statement']}", self.styles['Statement']),
                    f"${play['bearish_target']:.2f}",
                    f"${bearish_stop:.2f}",
                    self.format_rr_ratio(play['bearish_rr'])
                ])
                
                # Add separator row between different plays
                if len(group) > 1 and play.name != group.index[-1]:
                    table_data.append(['', '', '', '', '', ''])
            
            # Create and style table
            col_widths = [0.7*inch, 1.2*inch, 4.5*inch, 0.8*inch, 0.8*inch, 0.6*inch]
            table = Table(table_data, colWidths=col_widths)
            
            # Apply table styling
            style_commands = [
                # Header row
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3b82f6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                
                # All cells
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('ALIGN', (0, 1), (1, -1), 'CENTER'),
                ('ALIGN', (3, 1), (5, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                
                # Grid
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('LINEBELOW', (0, 0), (-1, 0), 1, colors.HexColor('#2563eb')),
                
                # Padding
                ('LEFTPADDING', (0, 0), (-1, -1), 4),
                ('RIGHTPADDING', (0, 0), (-1, -1), 4),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]
            
            # Color code R:R ratios
            for i in range(1, len(table_data)):
                if i % 3 == 1:  # Bullish rows
                    rr_value = table_data[i][5]
                    if rr_value != '' and ':' in rr_value:
                        rr_num = float(rr_value.split(':')[0])
                        if rr_num >= 2:
                            style_commands.append(('BACKGROUND', (5, i), (5, i), colors.HexColor('#d1fae5')))
                        elif rr_num >= 1:
                            style_commands.append(('BACKGROUND', (5, i), (5, i), colors.HexColor('#fef3c7')))
                elif i % 3 == 2:  # Bearish rows
                    rr_value = table_data[i][5]
                    if rr_value != '' and ':' in rr_value:
                        rr_num = float(rr_value.split(':')[0])
                        if rr_num >= 2:
                            style_commands.append(('BACKGROUND', (5, i), (5, i), colors.HexColor('#d1fae5')))
                        elif rr_num >= 1:
                            style_commands.append(('BACKGROUND', (5, i), (5, i), colors.HexColor('#fef3c7')))
            
            table.setStyle(TableStyle(style_commands))
            story.append(table)
            story.append(Spacer(1, 0.4*inch))
        
        # Page break before zone images
        story.append(PageBreak())
        
        # Section 2: Zone Images (one per page)
        section_header = Paragraph("Zone Charts", self.styles['CustomTitle'])
        story.append(section_header)
        story.append(PageBreak())
        
        # Add each image on its own page
        symbols_with_images = []
        for symbol in df['symbol'].unique():
            if symbol.upper() in images:
                symbols_with_images.append((symbol, images[symbol.upper()]))
        
        for idx, (symbol, image_path) in enumerate(symbols_with_images):
            # Symbol header
            symbol_header = Paragraph(f"{symbol} - Zone Chart", self.styles['SymbolHeader'])
            story.append(symbol_header)
            story.append(Spacer(1, 0.2*inch))
            
            try:
                # Resize image to fit page (larger since it's the only content)
                img_path, img_width, img_height = self.resize_image_for_pdf(
                    image_path, 
                    max_width=9*inch, 
                    max_height=6*inch
                )
                
                # Center the image
                img = Image(str(img_path), width=img_width, height=img_height)
                
                # Create a table to center the image
                img_table = Table([[img]], colWidths=[10*inch])
                img_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ]))
                
                story.append(img_table)
                
                # Zone info at bottom
                zone_data = grouped.get_group(symbol)
                price_lows = zone_data['price_low'].values
                price_highs = zone_data['price_high'].values
                zone_text = f"Zone Range: ${min(price_lows):.2f} - ${max(price_highs):.2f}"
                zone_info = Paragraph(zone_text, self.styles['ZoneInfo'])
                story.append(Spacer(1, 0.2*inch))
                story.append(zone_info)
                
            except Exception as e:
                logging.warning(f"Could not add image for {symbol}: {e}")
                error_text = Paragraph(f"Error loading image: {e}", self.styles['Normal'])
                story.append(error_text)
            
            # Add page break if not the last image
            if idx < len(symbols_with_images) - 1:
                story.append(PageBreak())
        
        # Add footer on last page
        story.append(Spacer(1, 0.5*inch))
        footer_text = "Generated by XIII Trading Systems - Daily Playbook Generator"
        footer = Paragraph(f"<font size='8' color='#9ca3af'><i>{footer_text}</i></font>", self.styles['Normal'])
        story.append(footer)
        
        # Build PDF
        doc.build(story)
        logging.info(f"PDF report generated: {output_path}")
        
    def run(self, dropbox_path: str, output_filename: str = None):
        """Generate the PDF report."""
        try:
            folder_path = Path(dropbox_path)
            if not folder_path.exists():
                raise ValueError(f"Folder path does not exist: {folder_path}")
                
            # Fetch today's plays
            logging.info(f"Fetching plays for {self.today}...")
            df = self.fetch_today_plays()
            
            # Find images
            logging.info("Scanning for zone images...")
            images = self.find_available_images(folder_path)
            
            # Generate output filename
            if not output_filename:
                output_filename = f"daily_playbook_report_{self.today.strftime('%Y%m%d')}.pdf"
            output_path = folder_path / output_filename
            
            # Generate PDF
            logging.info("Generating PDF report...")
            self.generate_pdf(df, images, output_path)
            
            print(f"\n{'='*60}")
            print(f"PDF REPORT GENERATED SUCCESSFULLY")
            print(f"{'='*60}")
            print(f"Output: {output_path}")
            print(f"Symbols: {df['symbol'].unique().tolist()}")
            print(f"Total plays: {len(df)}")
            print(f"{'='*60}\n")
            
            return str(output_path)
            
        except Exception as e:
            logging.error(f"Report generation failed: {e}")
            raise

def main():
    """Main entry point."""
    # Get Supabase credentials
    supabase_url = os.environ.get('SUPABASE_URL')
    supabase_key = os.environ.get('SUPABASE_KEY')
    
    if not supabase_url or not supabase_key:
        print("Error: SUPABASE_URL and SUPABASE_KEY not found")
        print("\nPlease ensure you have a .env file in your project root")
        sys.exit(1)
        
    # Get Dropbox path
    if len(sys.argv) > 1:
        dropbox_path = sys.argv[1]
    else:
        dropbox_path = input("Enter the Dropbox folder path: ").strip()
        
    # Generate report
    generator = DailyPlaybookReportGenerator(supabase_url, supabase_key)
    generator.run(dropbox_path)

if __name__ == "__main__":
    main()