"""
Text Processing Service
"""

from typing import List, Optional
from ..utils.file_parser import FileParser, split_text_into_chunks


class TextProcessor:
    """TextProcessor"""
    
    @staticmethod
    def extract_from_files(file_paths: List[str]) -> str:
        """从多个File提取Text"""
        return FileParser.extract_from_multiple(file_paths)
    
    @staticmethod
    def split_text(
        text: str,
        chunk_size: int = 500,
        overlap: int = 50
    ) -> List[str]:
        """
        分割Text
        
        Args:
            text: 原始Text
            chunk_size: 块大小
            overlap: 重叠大小
            
        Returns:
            Text块List
        """
        return split_text_into_chunks(text, chunk_size, overlap)
    
    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        预ProcessText
        - Remove多余空白
        - Standard化换行
        
        Args:
            text: 原始Text
            
        Returns:
            Process后的Text
        """
        import re
        
        # Standard化换行
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove连续空行(保留最多两个换行)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove行首行尾空白
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text.strip()
    
    @staticmethod
    def get_text_stats(text: str) -> dict:
        """GetTextStatisticsInfo"""
        return {
            "total_chars": len(text),
            "total_lines": text.count('\n') + 1,
            "total_words": len(text.split()),
        }

