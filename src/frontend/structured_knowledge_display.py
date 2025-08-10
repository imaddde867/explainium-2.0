"""
EXPLAINIUM - Structured Knowledge Display

Enhanced frontend component for displaying structured knowledge base output
from the AI Knowledge Analyst with clear organization and actionable insights.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional
import json
from datetime import datetime
import re


def display_structured_knowledge(knowledge_data: Dict[str, Any]) -> None:
    """
    Display structured knowledge analysis results with enhanced organization
    """
    if not knowledge_data:
        st.warning("No knowledge data to display")
        return
    
    # Check if this is the new structured format
    if knowledge_data.get('analysis_type') == 'structured_knowledge_analyst':
        display_ai_knowledge_analyst_results(knowledge_data)
    else:
        # Fallback to legacy display
        display_legacy_knowledge_results(knowledge_data)


def display_ai_knowledge_analyst_results(knowledge_data: Dict[str, Any]) -> None:
    """
    Display results from the AI Knowledge Analyst (3-phase framework)
    """
    st.title("🧠 AI Knowledge Analysis Report")
    
    # Document Context Overview
    if 'document_context' in knowledge_data:
        display_document_context(knowledge_data['document_context'])
    
    # Structured Output Section
    if 'structured_output' in knowledge_data:
        display_structured_output(knowledge_data['structured_output'])
    
    # Thematic Analysis
    if 'thematic_analysis' in knowledge_data:
        display_thematic_analysis(knowledge_data['thematic_analysis'])
    
    # Knowledge Items Table (for backward compatibility)
    if 'knowledge_items' in knowledge_data:
        display_knowledge_items_table(knowledge_data['knowledge_items'])


def display_document_context(context: Dict[str, Any]) -> None:
    """Display Phase 1 results: Document Context Overview"""
    st.header("📋 Document Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Document Type", context.get('type', 'Unknown').title())
        st.metric("Complexity", context.get('complexity', 'Unknown').title())
    
    with col2:
        st.metric("Domain", context.get('domain', 'Unknown').title())
        confidence = context.get('confidence', 0)
        st.metric("Analysis Confidence", f"{confidence:.1%}")
    
    with col3:
        st.metric("Target Audience", context.get('audience', 'Unknown').title())
    
    # Primary Purpose
    if context.get('purpose'):
        st.subheader("🎯 Primary Purpose")
        st.info(context['purpose'])


def display_structured_output(structured_output: Dict[str, Any]) -> None:
    """Display Phase 3 results: Synthesized Output"""
    
    # Executive Summary
    if structured_output.get('summary'):
        st.header("📊 Executive Summary")
        st.write(structured_output['summary'])
    
    # Key Takeaways
    if structured_output.get('key_takeaways'):
        st.header("🔑 Key Takeaways")
        for takeaway in structured_output['key_takeaways']:
            st.success(f"✅ {takeaway}")
    
    # Actionable Insights
    if structured_output.get('actionable_insights'):
        st.header("⚡ Actionable Insights")
        for insight in structured_output['actionable_insights']:
            st.warning(f"💡 {insight}")
    
    # Full Markdown Report
    if structured_output.get('markdown_report'):
        st.header("📄 Detailed Analysis Report")
        with st.expander("View Full Structured Report", expanded=False):
            st.markdown(structured_output['markdown_report'])


def display_thematic_analysis(thematic_analysis: Dict[str, Any]) -> None:
    """Display Phase 2 results: Thematic Analysis"""
    st.header("🏷️ Thematic Knowledge Categories")
    
    # Priority overview
    priority_counts = {}
    total_items = 0
    
    for category, data in thematic_analysis.items():
        priority = data.get('priority', 'informational')
        item_count = data.get('item_count', 0)
        priority_counts[priority] = priority_counts.get(priority, 0) + item_count
        total_items += item_count
    
    # Display priority distribution
    if priority_counts:
        st.subheader("📊 Priority Distribution")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Items", total_items)
        with col2:
            st.metric("Critical", priority_counts.get('critical', 0))
        with col3:
            st.metric("Important", priority_counts.get('important', 0))
        with col4:
            st.metric("Informational", priority_counts.get('informational', 0))
        
        # Priority chart
        if len(priority_counts) > 1:
            priority_df = pd.DataFrame([
                {'Priority': priority.title(), 'Count': count}
                for priority, count in priority_counts.items()
            ])
            
            fig = px.pie(priority_df, values='Count', names='Priority', 
                        title="Knowledge Items by Priority",
                        color_discrete_map={
                            'Critical': '#ff4444',
                            'Important': '#ffaa00', 
                            'Informational': '#4444ff'
                        })
            st.plotly_chart(fig, use_container_width=True)
    
    # Display each thematic category
    st.subheader("📂 Knowledge Categories")
    
    # Sort by priority
    priority_order = {'critical': 0, 'important': 1, 'informational': 2}
    sorted_categories = sorted(thematic_analysis.items(), 
                             key=lambda x: priority_order.get(x[1].get('priority', 'informational'), 3))
    
    for category, data in sorted_categories:
        display_thematic_category(category, data)


def display_thematic_category(category: str, data: Dict[str, Any]) -> None:
    """Display a single thematic category"""
    priority = data.get('priority', 'informational')
    item_count = data.get('item_count', 0)
    items = data.get('items', [])
    
    # Priority emoji
    priority_emoji = {'critical': '🔴', 'important': '🟡', 'informational': '🔵'}
    emoji = priority_emoji.get(priority, '⚪')
    
    # Category title
    category_title = category.replace('_', ' ').title()
    
    with st.expander(f"{emoji} {category_title} ({item_count} items - {priority.title()} Priority)", 
                     expanded=(priority == 'critical')):
        
        if not items:
            st.info("No items found in this category")
            return
        
        # Display items based on category type
        if category == 'processes_workflows':
            display_processes_workflows(items)
        elif category == 'policies_rules_requirements':
            display_policies_rules(items)
        elif category == 'key_data_metrics':
            display_key_data_metrics(items)
        elif category == 'roles_responsibilities':
            display_roles_responsibilities(items)
        elif category == 'definitions':
            display_definitions(items)
        elif category == 'risks_corrective_actions':
            display_risks_actions(items)
        else:
            # Generic display
            for i, item in enumerate(items, 1):
                st.write(f"{i}. {str(item)}")


def display_processes_workflows(items: List[Dict[str, Any]]) -> None:
    """Display processes and workflows with step-by-step formatting"""
    for i, item in enumerate(items, 1):
        if item.get('type') == 'workflow' and 'steps' in item:
            st.subheader(f"🔄 {item.get('title', f'Process {i}')}")
            
            for step in item['steps']:
                step_num = step.get('number', '?')
                description = step.get('description', 'No description')
                actionable = step.get('actionable', False)
                
                if actionable:
                    st.success(f"**Step {step_num}:** {description}")
                else:
                    st.info(f"**Step {step_num}:** {description}")
        else:
            st.write(f"🔄 {item.get('description', str(item))}")


def display_policies_rules(items: List[Dict[str, Any]]) -> None:
    """Display policies and requirements with emphasis on mandatory items"""
    for item in items:
        text = item.get('text', str(item))
        mandatory = item.get('mandatory', False)
        
        if mandatory:
            st.error(f"🚨 **MANDATORY:** {text}")
        else:
            st.info(f"📋 {text}")


def display_key_data_metrics(items: List[Dict[str, Any]]) -> None:
    """Display key data and metrics in organized format"""
    # Group by type
    by_type = {}
    for item in items:
        metric_type = item.get('type', 'general')
        if metric_type not in by_type:
            by_type[metric_type] = []
        by_type[metric_type].append(item)
    
    for metric_type, metrics in by_type.items():
        st.subheader(f"📊 {metric_type.title()}")
        
        for metric in metrics:
            value = metric.get('value', '')
            unit = metric.get('unit', '')
            context = metric.get('context', '')
            
            # Display as metric if it's a simple value
            if value and unit:
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric(f"{value} {unit}", "")
                with col2:
                    if context:
                        st.caption(context[:200] + "..." if len(context) > 200 else context)
            else:
                st.write(f"• {value} {unit} - {context}")


def display_roles_responsibilities(items: List[Dict[str, Any]]) -> None:
    """Display roles and responsibilities"""
    for item in items:
        role = item.get('role', 'Unknown Role')
        responsibility = item.get('responsibility', 'No description')
        
        st.write(f"👤 **{role}:** {responsibility}")


def display_definitions(items: List[Dict[str, Any]]) -> None:
    """Display definitions and terms"""
    for item in items:
        term = item.get('term', 'Unknown Term')
        definition = item.get('definition', 'No definition')
        
        with st.container():
            st.write(f"**{term}**")
            st.caption(definition)
            st.divider()


def display_risks_actions(items: List[Dict[str, Any]]) -> None:
    """Display risks and corrective actions"""
    for item in items:
        risk = item.get('risk', 'Unknown Risk')
        actions = item.get('corrective_actions', [])
        
        st.warning(f"⚠️ **Risk:** {risk}")
        
        if actions:
            st.write("**Corrective Actions:**")
            for action in actions:
                st.success(f"  ✅ {action}")
        
        st.divider()


def display_knowledge_items_table(knowledge_items: List[Dict[str, Any]]) -> None:
    """Display knowledge items in an enhanced table format"""
    if not knowledge_items:
        st.info("No knowledge items extracted")
        return
    
    st.header("📋 Detailed Knowledge Items")
    
    # Convert to DataFrame
    df = pd.DataFrame(knowledge_items)
    
    # Add filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Category filter
        categories = ['All'] + sorted(df['category'].unique().tolist()) if 'category' in df.columns else ['All']
        selected_category = st.selectbox("Filter by Category", categories)
    
    with col2:
        # Priority filter
        priorities = ['All'] + sorted(df['priority'].unique().tolist()) if 'priority' in df.columns else ['All']
        selected_priority = st.selectbox("Filter by Priority", priorities)
    
    with col3:
        # Confidence threshold
        min_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.5, 0.1)
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_category != 'All' and 'category' in df.columns:
        filtered_df = filtered_df[filtered_df['category'] == selected_category]
    
    if selected_priority != 'All' and 'priority' in df.columns:
        filtered_df = filtered_df[filtered_df['priority'] == selected_priority]
    
    if 'confidence' in df.columns:
        filtered_df = filtered_df[filtered_df['confidence'] >= min_confidence]
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Items", len(df))
    with col2:
        st.metric("Filtered Items", len(filtered_df))
    with col3:
        avg_confidence = filtered_df['confidence'].mean() if 'confidence' in filtered_df.columns and len(filtered_df) > 0 else 0
        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    with col4:
        critical_count = len(filtered_df[filtered_df['priority'] == 'critical']) if 'priority' in filtered_df.columns else 0
        st.metric("Critical Items", critical_count)
    
    # Display table
    if len(filtered_df) > 0:
        # Enhanced table display
        st.dataframe(
            filtered_df,
            use_container_width=True,
            height=400,
            column_config={
                'confidence': st.column_config.ProgressColumn(
                    "Confidence",
                    help="Confidence level of the extraction",
                    min_value=0.0,
                    max_value=1.0,
                ),
                'priority': st.column_config.SelectboxColumn(
                    "Priority",
                    help="Priority level of the knowledge item",
                    options=['critical', 'important', 'informational']
                )
            }
        )
        
        # Export options
        st.subheader("📤 Export Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_data = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download as CSV",
                data=csv_data,
                file_name=f"knowledge_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            json_data = filtered_df.to_json(orient='records', indent=2)
            st.download_button(
                label="Download as JSON",
                data=json_data,
                file_name=f"knowledge_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col3:
            if 'structured_output' in st.session_state.get('current_knowledge_data', {}):
                markdown_report = st.session_state['current_knowledge_data']['structured_output'].get('markdown_report', '')
                if markdown_report:
                    st.download_button(
                        label="Download Report (MD)",
                        data=markdown_report,
                        file_name=f"knowledge_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
    else:
        st.info("No items match the current filters")


def display_thematic_analysis(thematic_analysis: Dict[str, Any]) -> None:
    """Display Phase 2 results: Thematic Analysis with visual charts"""
    st.header("🏷️ Thematic Knowledge Analysis")
    
    # Create summary statistics
    category_data = []
    for category, data in thematic_analysis.items():
        category_data.append({
            'Category': category.replace('_', ' ').title(),
            'Items': data.get('item_count', 0),
            'Priority': data.get('priority', 'informational').title()
        })
    
    if category_data:
        # Category distribution chart
        df_categories = pd.DataFrame(category_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Items by category
            fig1 = px.bar(df_categories, x='Category', y='Items', 
                         title="Knowledge Items by Category",
                         color='Priority',
                         color_discrete_map={
                             'Critical': '#ff4444',
                             'Important': '#ffaa00',
                             'Informational': '#4444ff'
                         })
            fig1.update_xaxis(tickangle=45)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Priority distribution
            priority_summary = df_categories.groupby('Priority')['Items'].sum().reset_index()
            fig2 = px.pie(priority_summary, values='Items', names='Priority',
                         title="Items by Priority Level",
                         color_discrete_map={
                             'Critical': '#ff4444',
                             'Important': '#ffaa00',
                             'Informational': '#4444ff'
                         })
            st.plotly_chart(fig2, use_container_width=True)
    
    # Detailed category breakdown
    st.subheader("📂 Category Details")
    
    # Sort categories by priority
    priority_order = {'critical': 0, 'important': 1, 'informational': 2}
    sorted_categories = sorted(thematic_analysis.items(),
                             key=lambda x: priority_order.get(x[1].get('priority', 'informational'), 3))
    
    for category, data in sorted_categories:
        priority = data.get('priority', 'informational')
        item_count = data.get('item_count', 0)
        items = data.get('items', [])
        relationships = data.get('relationships', [])
        
        # Priority emoji
        priority_emoji = {'critical': '🔴', 'important': '🟡', 'informational': '🔵'}
        emoji = priority_emoji.get(priority, '⚪')
        
        category_title = category.replace('_', ' ').title()
        
        with st.expander(f"{emoji} {category_title} ({item_count} items)", 
                        expanded=(priority == 'critical')):
            
            if items:
                # Display category-specific content
                if category == 'processes_workflows':
                    display_processes_workflows(items)
                elif category == 'policies_rules_requirements':
                    display_policies_rules(items)
                elif category == 'key_data_metrics':
                    display_key_data_metrics(items)
                elif category == 'roles_responsibilities':
                    display_roles_responsibilities(items)
                elif category == 'definitions':
                    display_definitions(items)
                elif category == 'risks_corrective_actions':
                    display_risks_actions(items)
                
                # Show relationships
                if relationships:
                    st.caption(f"**Relationships:** {', '.join(relationships)}")
            else:
                st.info("No items found in this category")


def display_legacy_knowledge_results(knowledge_data: Dict[str, Any]) -> None:
    """Display legacy knowledge extraction results"""
    st.title("📊 Knowledge Extraction Results")
    
    # Handle legacy format
    if 'knowledge_items' in knowledge_data:
        knowledge_items = knowledge_data['knowledge_items']
    elif 'passes' in knowledge_data:
        # Convert legacy passes format
        knowledge_items = convert_legacy_passes_to_items(knowledge_data['passes'])
    else:
        st.warning("No recognized knowledge data format")
        return
    
    display_knowledge_items_table(knowledge_items)


def convert_legacy_passes_to_items(passes: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert legacy passes format to knowledge items"""
    items = []
    
    for pass_name, pass_data in passes.items():
        if isinstance(pass_data, dict) and 'concepts' in pass_data:
            for concept in pass_data['concepts']:
                items.append({
                    'type': pass_name,
                    'category': pass_name.title(),
                    'content': str(concept),
                    'priority': 'informational',
                    'confidence': 0.8,
                    'metadata': concept
                })
    
    return items


def create_knowledge_visualization(knowledge_items: List[Dict[str, Any]]) -> None:
    """Create enhanced visualizations for knowledge items"""
    if not knowledge_items:
        return
    
    df = pd.DataFrame(knowledge_items)
    
    st.subheader("📈 Knowledge Analytics")
    
    # Distribution charts
    col1, col2 = st.columns(2)
    
    with col1:
        if 'category' in df.columns:
            category_counts = df['category'].value_counts()
            fig1 = px.bar(x=category_counts.index, y=category_counts.values,
                         title="Knowledge Items by Category")
            st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        if 'confidence' in df.columns:
            fig2 = px.histogram(df, x='confidence', nbins=10,
                              title="Confidence Distribution")
            st.plotly_chart(fig2, use_container_width=True)


def display_search_and_filter_interface(knowledge_items: List[Dict[str, Any]]) -> pd.DataFrame:
    """Display search and filter interface for knowledge items"""
    if not knowledge_items:
        return pd.DataFrame()
    
    df = pd.DataFrame(knowledge_items)
    
    st.subheader("🔍 Search & Filter")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Search box
        search_term = st.text_input("Search knowledge items", "")
    
    with col2:
        # Quick filters
        show_high_confidence = st.checkbox("High Confidence Only (>80%)", False)
    
    # Apply filters
    filtered_df = df.copy()
    
    if search_term:
        # Search in all text columns
        text_columns = [col for col in df.columns if df[col].dtype == 'object']
        mask = df[text_columns].astype(str).apply(
            lambda x: x.str.contains(search_term, case=False, na=False)
        ).any(axis=1)
        filtered_df = filtered_df[mask]
    
    if show_high_confidence and 'confidence' in df.columns:
        filtered_df = filtered_df[filtered_df['confidence'] > 0.8]
    
    return filtered_df


# Helper function to store current knowledge data for exports
def store_knowledge_data(knowledge_data: Dict[str, Any]) -> None:
    """Store knowledge data in session state for exports"""
    st.session_state['current_knowledge_data'] = knowledge_data


# Main display function to be called from the main app
def render_structured_knowledge_interface(knowledge_data: Dict[str, Any]) -> None:
    """Main function to render the structured knowledge interface"""
    # Store data for exports
    store_knowledge_data(knowledge_data)
    
    # Display the structured knowledge
    display_structured_knowledge(knowledge_data)
    
    # Add visualizations if we have knowledge items
    if knowledge_data.get('knowledge_items'):
        create_knowledge_visualization(knowledge_data['knowledge_items'])
