from pydantic import BaseModel, Field
from typing_extensions import Literal


class ClassificationModel(BaseModel):
    """Analyze the message and route it according to its content."""

    classification: Literal["legal", "support", "unsafe", "unknown"] = Field(
        description="""The classification of the input: 'set to 'legal' if the input is a query related to legal, 'support' if related to software or technical support, or
        'unsafe' if the input fails the moderation/safety check, and 'unknown' for everything else.
        Examples of legal questions that can be processed:
        - questions about various software licenses
        - embargoes for certain types of software that prevent delivery to various countries outside the United States
        - privacy, access restrictions, around customer data (sometimes referred to as PII)
        - questions about contracts, policies, procedures, or compliance
        Examples of support (software support, technical support, IT support) that can be processed:
        - the user cites problems running certain applications of the company's OpenShift Cluster
        - the user asks to have new applications deployed on the company's OpenShift Cluster
        - the user needs permissions to access certain resources on the company's OpenShift Cluster
        - the user asks about current utilization of resources on the company's OpenShift Cluster
        - the user cites issues with performance of their application specifically or the OpenShift Cluster in general
        - questions about FantaCo products like CloudSync, TechGear Pro Laptop
        - installation, setup, or configuration questions for software or hardware
        - troubleshooting issues with devices, drivers, or applications
        - syncing issues, file transfer problems, or connectivity questions
        - any technical how-to questions about products or systems
        """,
    )


class SupportClassificationModel(BaseModel):
    """Analyze the message and route it according to its content."""

    classification: Literal["pod", "perf", "git"] = Field(
        description="""
        The classification of the input: set the classification to 'perf' if there is any mention of
        - performance
        - the application is slow to respond
        - questions around CPU or memory consumption or usage
        However, set the classification to 'pod' if the input asks for
        - assistance with an application, and
        - makes any reference to a 'Namespace' or 'Project' that exists within OpenShift or Kubernetes

        Otherwise, set the classification to 'git'.
        """,
    )
    namespace: str = Field(
        description="""
        the namespace of the input: if the query makes any reference to a namespace or project of a given name, then set the
        namespace field here to the first given name referenced as a namespace or project.
        """,
    )
    performance: str = Field(
        description="""
        if the query makes any reference to performance, applications running slowly, CPU or memory utilization or consumption, then
        set the performance field to 'true'.  Otherwise, if there is no mentioned of performance, being slow, CPU or memory,
        set the performance field to 'false' or an empty string.
        """,
    )
